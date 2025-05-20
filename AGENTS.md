# AGENTS Instructions

This repository contains a simple voice interface for Claude Code.

## Setup
- Copy `.env.sample` to `.env` and provide your `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.
- The recommended way to run scripts is with the `uv` package manager.

## Running the assistant
- Launch the assistant with `uv run voice_to_claude_code.py`.
- Optional arguments include `--id` to resume a conversation and `--prompt` to process an initial prompt.

## Tests
- Run `pytest -q` to execute the unit tests. Tests are skipped unless the `claude` CLI is installed and `RUN_CLAUDE_TESTS=1` is set.
- Please run the test suite before committing any changes.

## Coding style
- Follow standard PEP8 formatting. Use clear names and add docstrings for functions or classes.
- Organize imports into standard library, third party, and local sections.
- Keep functions small and focused.

## Documentation
- Update `README.md` when introducing new functionality or commands.

