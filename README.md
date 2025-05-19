# Claude Code Voice Agent

This repository provides a voice-enabled interface for [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) and the [Claude API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api).

The system listens for a trigger word (default: `"Athena"`) and sends your spoken request to either Claude Code CLI or the Claude API, then plays back the response using OpenAI's text-to-speech.

This project began as a fork of [`disler/claude-code-is-programmable`](https://github.com/disler/claude-code-is-programmable) by [@disler](https://github.com/disler). It has been expanded to include both Claude Code CLI and Claude API interfaces.

![Voice to Claude Code](images/voice-to-claude-code.png)

## Features

- 🎤 **Voice interaction** with Claude using RealtimeSTT for speech recognition
- 🔄 **Dual implementations**: Use Claude Code CLI for coding tasks or Claude API for general queries
- 🗣️ **Natural responses** with OpenAI TTS for speech output
- 🧠 **Conversation memory** with YAML-based conversation history
- 🔧 **Robust error handling** with model fallbacks and recovery mechanisms
- 📝 **Text-only mode** for terminal-based interactions

## Quick Start

```bash
# Make scripts executable if needed
chmod +x voice_to_claude_code.py voice_to_claude_api.py

# Run API version (recommended for general questions)
./voice_to_claude_api.py

# Run CLI version (recommended for code tasks)
./voice_to_claude_code.py

# Run in text-only mode (no voice input/output)
./voice_to_claude_code.py --text-only

# Run with a specific prompt
./voice_to_claude_api.py --prompt "Athena, tell me a joke"
```

## Setup

1. Clone the repository and navigate to the directory
2. Copy `.env.sample` to `.env` and add your `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`
3. Install [uv](https://github.com/astral-sh/uv) for running Python scripts with pinned dependencies
4. Run the audio device test utilities to configure your microphone and speaker

```bash
uv run test_mic.py    # Test and configure your microphone
uv run test_speaker.py  # Test and configure your speaker
```

## Usage Options

You can use either the Claude API version or the Claude Code CLI version:

### Claude API Version (recommended for general questions)

```bash
# Run with default settings
uv run voice_to_claude_api.py

# Resume a previous conversation
uv run voice_to_claude_api.py --id my-chat-id

# Start with an initial prompt
uv run voice_to_claude_api.py --prompt "Athena, what's the weather like today?"
```

### Claude Code CLI Version (recommended for coding tasks)

```bash
# Run with default settings
uv run voice_to_claude_code.py

# Use text-only mode (no voice)
uv run voice_to_claude_code.py --text-only

# Start with a coding task
uv run voice_to_claude_code.py --prompt "Athena, write a Python function to calculate Fibonacci numbers"
```

## Tests

The repository includes various test utilities:

- `test_mic.py`: Test and configure your microphone
- `test_speaker.py`: Test and configure your speaker
- `test_stt.py`: Test speech-to-text functionality
- Tests for Claude integration (in the `tests` directory)

Run standard tests with:

```bash
uv run pytest -q
```

## Troubleshooting

See the [CLAUDE.md](CLAUDE.md) file for detailed troubleshooting steps and advanced configuration options.
