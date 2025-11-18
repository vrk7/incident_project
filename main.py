"""Incident & Outage Postmortem Assistant CLI."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = """You are an SRE that writes concise, structured incident postmortems.
Given raw logs, chat transcripts, and ticket descriptions, distill the most
important data. Always respond with valid JSON that matches this schema:
{
  "timeline": [
    {"timestamp": "ISO-8601 or 'unknown'", "description": "event"}
  ],
  "root_causes": [
    {
      "description": "hypothesis",
      "confidence": "high|medium|low",
      "evidence": "brief supporting points"
    }
  ],
  "action_items": [
    {
      "description": "action",
      "priority": "P1|P2|P3",
      "category": "monitoring|process|code|infra|runbook|other"
    }
  ],
  "summary": "2-3 sentence summary of the incident",
  "impact": "Impact scope, duration, customers, etc."
}
Keep answers practical and consistent with SRE best practices.
"""

TIMESTAMP_REGEXES: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?\b"),
    re.compile(r"\b\d{2}:\d{2}:\d{2}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
)

ISO_FALLBACK_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
)


def resolve_env_value(
    direct_value: str | None,
    env_var: str | None,
    *,
    required: bool,
    missing_message: str,
) -> str | None:
    """Resolve configuration from either a direct CLI value or an env var name."""

    if direct_value:
        return direct_value

    if env_var:
        env_value = os.getenv(env_var)
        if env_value:
            return env_value

    if required:
        raise EnvironmentError(missing_message)

    return None


def build_llm(
    *, model: str = "gpt-4o-mini", temperature: float = 0.1, api_key: str, base_url: str | None
) -> ChatOpenAI:
    """Create the LangChain ChatOpenAI client."""

    if not api_key:
        raise EnvironmentError("Missing OpenAI API key. Provide --api-key or set the env var.")

    client_kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        client_kwargs["base_url"] = base_url

    return ChatOpenAI(**client_kwargs)


def extract_json_block(text: str) -> Dict[str, Any]:
    """Extract the first JSON object found inside a string."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("Could not locate JSON object in LLM response.")
    json_text = text[start : end + 1]
    return json.loads(json_text)


def truncate_text(text: str, limit: int | None) -> Tuple[str, bool]:
    """Clip large artifacts to avoid blowing past context limits."""

    if not limit or limit <= 0 or len(text) <= limit:
        return text, False

    head = int(limit * 0.6)
    tail = limit - head
    truncated = text[:head].rstrip() + "\n...\n" + text[-tail:].lstrip()
    return truncated, True


def _extract_timeline_hints(text: str, source: str, limit: int) -> List[Dict[str, str]]:
    """Pull lightweight timestamp + context pairs from a blob of text."""

    if limit <= 0:
        return []
    hints: List[Dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        timestamp = None
        for regex in TIMESTAMP_REGEXES:
            match = regex.search(line)
            if match:
                timestamp = match.group(0)
                break
        if not timestamp:
            continue
        hints.append(
            {
                "source": source,
                "timestamp": timestamp,
                "context": line[:240],
            }
        )
        if len(hints) >= limit:
            break
    return hints


def build_timeline_hints(sources: Iterable[Tuple[str, str]], limit: int) -> List[Dict[str, str]]:
    """Combine hints from multiple artifacts, capped to a global limit."""

    if limit <= 0:
        return []
    source_list = list(sources)
    if not source_list:
        return []
    per_source = max(1, limit // len(source_list))
    hints: List[Dict[str, str]] = []
    for source, text in source_list:
        hints.extend(_extract_timeline_hints(text, source, per_source))
    return hints[:limit]


def parse_timestamp_value(ts: str) -> datetime | None:
    """Best-effort parsing for timeline timestamps."""

    if not ts or ts.lower() == "unknown":
        return None
    ts = ts.strip()
    normalized = ts
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ISO_FALLBACK_FORMATS:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def summarize_timeline_span(timeline: Sequence[Dict[str, Any]]) -> tuple[datetime, datetime] | None:
    """Return first+last timestamps when enough structured data exists."""

    parsed = [parse_timestamp_value(event.get("timestamp", "")) for event in timeline]
    filtered = [dt for dt in parsed if dt]
    if len(filtered) < 2:
        return None
    filtered.sort()
    return filtered[0], filtered[-1]


def format_duration(delta: timedelta) -> str:
    """Human friendly duration formatting."""

    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "0s"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds and not hours:
        # keep seconds when short incidents
        parts.append(f"{seconds}s")
    return " ".join(parts) or "0s"


def run_analysis(
    logs: str,
    chat: str,
    ticket: str,
    model: str,
    *,
    api_key: str,
    api_base: str | None,
    timeline_hints: Sequence[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Use the LLM to analyze artifacts and return structured insights."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                (
                    "Logs:\n{logs}\n\n"
                    "Chat transcript:\n{chat}\n\n"
                    "Ticket description:\n{ticket}\n\n"
                    "Timeline hints (heuristic):\n{timeline_hints}\n\n"
                    "Summarize the incident strictly using JSON as specified earlier."
                ),
            ),
        ]
    )

    llm = build_llm(model=model, api_key=api_key, base_url=api_base)
    response = (prompt | llm).invoke(
        {
            "logs": logs,
            "chat": chat,
            "ticket": ticket,
            "timeline_hints": json.dumps(timeline_hints or [], ensure_ascii=False),
        }
    )
    return extract_json_block(response.content)


def build_postmortem(data: Dict[str, Any]) -> str:
    """Create the Markdown postmortem document from structured data."""

    timeline_lines = []
    for event in data.get("timeline", []):
        ts = event.get("timestamp", "unknown")
        desc = event.get("description", "")
        timeline_lines.append(f"- **{ts}** – {desc}")

    root_cause_lines = []
    for rc in data.get("root_causes", []):
        desc = rc.get("description", "")
        conf = rc.get("confidence", "unknown")
        evidence = rc.get("evidence", "")
        root_cause_lines.append(f"- **{desc}** (confidence: {conf})\n  - Evidence: {evidence}")

    action_lines = []
    for action in data.get("action_items", []):
        desc = action.get("description", "")
        priority = action.get("priority", "P3")
        category = action.get("category", "other")
        action_lines.append(f"- **{desc}** *(priority: {priority}, category: {category})*")

    return "\n".join(
        [
            "# Incident Postmortem",
            "",
            "## Summary",
            data.get("summary", ""),
            "",
            "## Impact",
            data.get("impact", ""),
            "",
            "## Root Cause Hypotheses",
            "\n".join(root_cause_lines) or "- Not enough information yet.",
            "",
            "## Timeline",
            "\n".join(timeline_lines) or "- Timeline unavailable.",
            "",
            "## Action Items",
            "\n".join(action_lines) or "- No action items generated.",
        ]
    )


def read_file(path: Path) -> str:
    """Read UTF-8 text from disk."""

    return path.read_text(encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Configure the CLI arguments."""

    parser = argparse.ArgumentParser(description="Incident & Outage Postmortem Assistant")
    parser.add_argument("--logs", required=True, type=Path, help="Path to logs text file")
    parser.add_argument("--chat", required=True, type=Path, help="Path to chat transcript file")
    parser.add_argument("--ticket", required=True, type=Path, help="Path to incident ticket file")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat completion model")
    parser.add_argument(
        "--output", default=Path("postmortem.md"), type=Path, help="Output markdown file"
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to save the structured JSON analysis alongside the markdown",
    )
    parser.add_argument(
        "--max-artifact-chars",
        type=int,
        default=15000,
        help="Trim each artifact to this many characters to stay within context limits (<=0 disables)",
    )
    parser.add_argument(
        "--timeline-hints-limit",
        type=int,
        default=30,
        help="Max heuristic hints extracted from logs/chats to guide the LLM's timeline",
    )
    parser.add_argument("--api-key", help="Direct OpenAI-compatible API key")
    parser.add_argument(
        "--api-key-var",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument("--api-base", help="Direct override for the OpenAI-compatible base URL")
    parser.add_argument(
        "--api-base-var",
        default="OPENAI_API_BASE",
        help="Environment variable that stores a custom base URL (optional)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    raw_logs = read_file(args.logs)
    raw_chat = read_file(args.chat)
    raw_ticket = read_file(args.ticket)

    logs_text, logs_truncated = truncate_text(raw_logs, args.max_artifact_chars)
    chat_text, chat_truncated = truncate_text(raw_chat, args.max_artifact_chars)
    ticket_text, ticket_truncated = truncate_text(raw_ticket, args.max_artifact_chars)
    timeline_hints = build_timeline_hints(
        [("logs", logs_text), ("chat", chat_text)], limit=args.timeline_hints_limit
    )

    api_key = resolve_env_value(
        args.api_key,
        args.api_key_var,
        required=True,
        missing_message=(
            "Provide an API key via --api-key or set the environment variable "
            f"{args.api_key_var}."
        ),
    )
    api_base = resolve_env_value(
        args.api_base,
        args.api_base_var,
        required=False,
        missing_message="",
    )

    analysis = run_analysis(
        logs_text,
        chat_text,
        ticket_text,
        args.model,
        api_key=api_key,
        api_base=api_base,
        timeline_hints=timeline_hints,
    )
    markdown = build_postmortem(analysis)
    args.output.write_text(markdown, encoding="utf-8")
    if args.json_output:
        args.json_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    timeline_count = len(analysis.get("timeline", []))
    root_cause_count = len(analysis.get("root_causes", []))
    action_count = len(analysis.get("action_items", []))
    span = summarize_timeline_span(analysis.get("timeline", []))

    summary_parts = [
        f"Generated {args.output} with ",
        f"{timeline_count} timeline events, ",
        f"{root_cause_count} root cause hypotheses, and ",
        f"{action_count} action items.",
    ]
    if span:
        start, end = span
        duration = end - start
        summary_parts.append(
            f" Approximate incident window: {start.isoformat()} → {end.isoformat()} "
            f"({format_duration(duration)})."
        )
    print("".join(summary_parts))

    truncated_sections = [
        name
        for name, flag in (
            ("logs", logs_truncated),
            ("chat", chat_truncated),
            ("ticket", ticket_truncated),
        )
        if flag
    ]
    if truncated_sections:
        print(
            "Note: truncated "
            + ", ".join(truncated_sections)
            + f" to {args.max_artifact_chars} characters to fit into the prompt."
        )
    if args.json_output:
        print(f"Structured analysis JSON saved to {args.json_output}.")


if __name__ == "__main__":
    main()
