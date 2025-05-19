# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository provides a voice-enabled interface for Claude Code, allowing users to interact with Claude through voice commands. The system uses speech-to-text (RealtimeSTT) to listen for a trigger word ("Athena"), processes the request through Claude Code or the Claude API, and plays back responses using OpenAI's text-to-speech.

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

## Environment Setup

- Use UV for package management and running Python scripts:
  ```bash
  # Install dependencies and run scripts with UV
  uv run voice_to_claude_code.py
  
  # Run with specific conversation ID (to continue previous conversations)
  uv run voice_to_claude_code.py --id my-chat-id
  
  # Run with initial prompt
  uv run voice_to_claude_code.py --prompt "Athena, create a hello world script"
  
  # Run tests
  uv run pytest -q
  ```

## Key Components

1. **voice_to_claude_code.py**: Main script that powers the voice interface to Claude Code CLI
   - Listens for the trigger word "Athena"
   - Sends the transcribed speech to Claude Code CLI
   - Processes the response through TTS for speech output
   - Best for coding tasks as it leverages Claude Code tools

2. **voice_to_claude_api.py**: Alternative implementation that uses the Anthropic API directly
   - Similar functionality but calls the Claude API directly
   - Includes model fallbacks if the primary model fails
   - Best for general questions and conversation

3. **claude_testing_v1.py**: Helper utilities for running Claude CLI in headless mode
   - Provides wrapper functions to run Claude in test environments

4. **Test utilities**:
   - test_mic.py: Test microphone input configuration
   - test_speaker.py: Test speaker output configuration
   - test_stt.py: Test speech-to-text functionality

## Key Configuration Settings

The main scripts contain several configurable parameters:
- TRIGGER_WORDS: List of trigger phrases that activate Claude (default: "athena")
- STT_MODEL: Speech-to-text model to use ("tiny.en", "small.en", etc.)
- TTS_VOICE: Text-to-speech voice to use (OpenAI voices)
- INPUT/OUTPUT_DEVICE: Audio device configuration
- CLAUDE_MODEL: Claude API model version (when using API version)
- DEFAULT_CLAUDE_TOOLS: Tools enabled for Claude Code CLI

## Development Workflow

1. **Setting up the environment**:
   ```bash
   # Clone the repository
   git clone https://github.com/lee-geyer/claude-code-voice-agent.git
   cd claude-code-voice-agent
   
   # Copy environment variables template and add your API keys
   cp .env.sample .env
   # Edit .env to add your API keys
   ```

2. **Testing audio devices**:
   ```bash
   uv run test_mic.py  # Identify the correct microphone device index
   uv run test_speaker.py  # Identify the correct speaker device index
   ```

3. **Running tests**:
   ```bash
   # Run all tests with pytest
   uv run pytest -q
   
   # Run Claude-specific tests (requires RUN_CLAUDE_TESTS=1)
   RUN_CLAUDE_TESTS=1 uv run pytest -q tests/test_claude_testing_v1.py
   ```

## Project Architecture

The system has two main variants:
1. **CLI-based (voice_to_claude_code.py)**: Uses the Claude CLI tool to interact with Claude Code
   - More capable for coding tasks
   - Requires Claude CLI to be installed
   - Can use all Claude Code tools (Bash, Edit, Write, etc.)

2. **API-based (voice_to_claude_api.py)**: Uses the Anthropic API directly
   - More reliable for general knowledge questions
   - Doesn't require Claude CLI
   - Has fallback models if one fails
   - Easier to set up (just needs API key)

Both implementations share common components:
- Speech recognition with RealtimeSTT
- Word-level trigger word detection
- Response compression for better voice output
- Text-to-speech with OpenAI TTS
- Conversation history management in YAML files
- Recovery mechanisms for audio device issues

## Common Troubleshooting

- **Audio device issues**: Run the test_mic.py and test_speaker.py utilities to identify correct device indices
- **Missing API keys**: Ensure both ANTHROPIC_API_KEY and OPENAI_API_KEY are properly set in your .env file
- **Claude CLI not available**: The voice_to_claude_code.py requires Claude CLI to be installed
- **Trigger word not detected**: Speak clearly and ensure "Athena" is a distinct word in your request
- **API errors**: If you see model not found errors, check that your API key has access to the models
- **Empty responses**: Try restarting the application or check your internet connection
- **Speech recognition issues**: Try speaking more slowly and clearly near the microphone