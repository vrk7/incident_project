# Incident Postmortem Assistant

A small LangChain-powered command-line tool that turns raw incident artifacts
(logs, chat transcripts, and ticket descriptions) into a structured Markdown
postmortem.

## Requirements

- Python 3.10+
- Node.js 18+ (so you can drive the CLI via `npm` or `pnpm` scripts)
- Dependencies listed in `requirements.txt`
- An LLM API key available via an environment variable (defaults to `OPENAI_API_KEY`)

Install the dependencies with whichever tooling you prefer:

```bash
# Python-only workflow
pip install -r requirements.txt

# or use the provided Node scripts
pnpm install   # or `npm install` (no JS deps, but sets up lock files)
pnpm run setup # runs `python -m pip install -r requirements.txt`
```

## Usage

1. Prepare three UTF-8 text files: one with relevant logs, one with a chat
   transcript (Slack, Teams, etc.), and one with the ticket/alert description.
2. Run the CLI:

```bash
python3 main.py --logs logs.txt --chat slack.txt --ticket ticket.txt
# or
pnpm run postmortem -- --logs logs.txt --chat slack.txt --ticket ticket.txt
```

Optional flags:

- `--model`: override the default `gpt-4o-mini` model name.
- `--output`: change the destination file (defaults to `postmortem.md`).
- `--json-output`: write the structured JSON that the LLM produced to another file for downstream tooling.
- `--max-artifact-chars`: clip each artifact to the specified number of characters to prevent oversized prompts.
- `--timeline-hints-limit`: number of heuristic timestamp hints (from logs/chats) forwarded to the LLM to improve the timeline.
- `--api-key`: pass an API key directly instead of via environment variable.
- `--api-key-var`: name of the env var that stores the API key (defaults to `OPENAI_API_KEY`).
- `--api-base`: override the OpenAI-compatible base URL directly.
- `--api-base-var`: name of the env var that stores a custom base URL (defaults to `OPENAI_API_BASE`).
- `--env-file`: path to a `.env`-style file that the CLI should load before resolving variables (defaults to `.env`).
- `--skip-env-file`: disable `.env` loading entirely for highly locked-down environments.

After execution you will see a short console summary, and the complete draft
postmortem will be saved to `postmortem.md`.

## What each file does (in simple words)

- `main.py`: the actual assistant. It reads the three input files, asks the LLM
  (via LangChain + OpenAI) for a structured analysis, converts it into a
  Markdown report, and prints a quick summary to the console.
- `requirements.txt`: the tiny list of Python libraries you need to install to
  run the tool (LangChain core + the OpenAI chat wrapper).
- `package.json`: lets you call the assistant with familiar `npm run`/`pnpm run`
  scripts (for local dev shells or CI pipelines) and documents the supported
  Node version.
- `.env.example`: template that shows which environment variables the CLI will
  load automatically. Copy it to `.env`, customize the secrets, and you are
  ready to go.
- `README.md`: the document you are reading now—it explains how to install,
  run, and extend the assistant.

## Environment variables

You can keep your runtime credentials in separate environment variables to match the
deployment you are integrating with:

| Purpose | Default variable | Notes |
| --- | --- | --- |
| API key | `OPENAI_API_KEY` | Required. Change the var name via `--api-key-var` or provide `--api-key`. |
| Base URL | `OPENAI_API_BASE` | Optional. Useful for Azure OpenAI or compatible gateways. |

Because the assistant lets you pick which variables to use, teams can maintain
environment-specific `.env` files (for prod vs. staging) or inject secrets from
their orchestrator without modifying the code. The CLI attempts to load `.env`
from the working directory (or whichever file you pass via `--env-file`) before
inspecting the live environment, so developers can keep secrets outside of shell
history yet still drive the tool with a single `pnpm run postmortem -- ...`.

## How it works

The CLI builds a single prompt containing the logs, chat, and ticket content and
asks the model for a JSON description that includes:

- A timeline of important events
- Root-cause hypotheses (with confidence + evidence)
- Action items (with priority + category)
- Summary and impact sections

Before the prompt is sent, the assistant now performs a couple of quality-of-life
steps:

1. **Artifact trimming** – each file is clipped to a configurable length (keep
   it at `<=0` to disable) so large runbooks don’t blow past the context window.
2. **Timeline hints** – timestamps spotted in logs and chats are forwarded to
   the model as JSON hints, which dramatically improves chronological accuracy.

The JSON response is converted into Markdown with dedicated sections, and you
can optionally persist that JSON via `--json-output`. Because it uses
LangChain’s `ChatPromptTemplate` + `ChatOpenAI`, swapping to a different
OpenAI-compatible model is just a CLI flag change.

## Integrating into a real-time workflow

To plug the assistant into a real-time incident pipeline:

1. Hook it into your incident-bot or paging workflow so that, once an incident
   is resolved, the bot dumps relevant logs, chat transcripts, and ticket text
   to temporary files.
2. Invoke the CLI programmatically (e.g., via a small wrapper script or an
   orchestration tool like Airflow/GitHub Actions) and capture the generated
   `postmortem.md`.
3. Publish the Markdown directly to your wiki, ticketing system, or status page
   using your existing APIs.
4. (Optional) Keep the generated JSON payload (before Markdown conversion) to
   feed downstream analytics—pass `--json-output analysis.json` to save it, or
   forward the file to your observability stack.
Because everything is local and file-based, embedding this command into chatops
slash-commands or CI pipelines only requires mounting the three text files and
passing their paths to `python main.py`.
