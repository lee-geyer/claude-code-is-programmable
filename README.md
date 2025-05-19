# Claude Code Voice Agent

This repository provides a simple voice-enabled interface for [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview).

The script `voice_to_claude_code.py` listens for a trigger word (default `"claude"`) and sends your spoken request to Claude Code, then plays the response using OpenAI TTS.

This project is forked from [indydevdan's `claude-code-is-programmable`](https://github.com/indydevdan/claude-code-is-programmable). It was trimmed using OpenAI Codex to focus on the voice agent for Claude Code.

![Voice to Claude Code](images/voice-to-claude-code.png)

## Setup

1. Copy `.env.sample` to `.env` and add your `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.
2. Install [uv](https://github.com/astral-sh/uv) for running Python scripts with pinned dependencies.

## Usage

Run the assistant:

```bash
uv run voice_to_claude_code.py
```

You can resume a previous conversation or provide an initial prompt:

```bash
uv run voice_to_claude_code.py --id my-chat-id
uv run voice_to_claude_code.py --prompt "create a hello world script"
```

Speak a request that includes the trigger word and the assistant will respond through your speakers.

## Tests

The `tests` directory includes a small validation of the `run_claude_json` helper. Tests are skipped unless the `claude` CLI is available and `RUN_CLAUDE_TESTS=1` is set.

Run tests with:

```bash
pytest -q
```
