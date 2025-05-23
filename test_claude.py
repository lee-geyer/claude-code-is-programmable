#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "rich",
#   "python-dotenv",
# ]
# ///

import os
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API keys
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY not found in environment variables")
    exit(1)

# Set up rich console
console = Console()

def run_claude_code(prompt):
    """Run Claude Code with the given prompt"""
    console.print(f"\n[bold blue]Running Claude Code with prompt:[/bold blue]")
    console.print(Panel(prompt))
    
    # Claude CLI command
    cmd = [
        "/Users/leegeyer/.claude/local/claude",
        "-p",
        prompt,
        "--allowedTools",
        "Bash", "Edit", "Write", "Glob", "Grep", "LS", "Read"
    ]
    
    try:
        # Execute Claude Code
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Get the response
        response = process.stdout
        
        # Display the response
        console.print(Panel(title="Claude Response", renderable=Markdown(response)))
        return response
        
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error running Claude Code:[/bold red]")
        console.print(f"Exit code: {e.returncode}")
        console.print(f"Error: {e.stderr[:500]}...")
        return None

def main():
    console.print(Panel.fit(
        "[bold green]Claude Code Test Script[/bold green]\n"
        "This script tests the Claude Code CLI integration."
    ))
    
    # Simple test prompt
    test_prompt = """
    # Test Prompt
    
    You are Claude, an AI assistant. Please respond to this test query to confirm
    that the Claude Code CLI integration is working properly.
    
    Tell me what files are in the current directory.
    """
    
    response = run_claude_code(test_prompt)
    
    if response:
        console.print("[bold green]✓ Claude Code integration test successful![/bold green]")
    else:
        console.print("[bold red]✗ Claude Code integration test failed.[/bold red]")

if __name__ == "__main__":
    main()