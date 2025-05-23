#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "rich",
#   "python-dotenv",
#   "anthropic",
# ]
# ///

import os
import sys
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Check for API keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("Error: ANTHROPIC_API_KEY not found in environment variables")
    exit(1)

# Set up rich console
console = Console()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def run_claude_request(prompt):
    """Run a direct API call to Claude with the given prompt"""
    console.print(f"\n[bold blue]Sending request to Claude:[/bold blue]")
    console.print(Panel(prompt))
    
    try:
        # Format messages
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Send request to Claude using the latest model
        response = client.messages.create(
            model="claude-3-sonnet-20240307", # Updated to the latest stable model
            max_tokens=1024,
            temperature=0.3,
            messages=messages,
        )
        
        # Get the response text
        response_text = response.content[0].text
        
        # Display the response
        console.print(Panel(title="Claude Response", renderable=Markdown(response_text)))
        return response_text
        
    except Exception as e:
        console.print(f"[bold red]Error calling Claude API:[/bold red]")
        console.print(f"Error: {str(e)}")
        return None

def run_claude_with_tool_use(prompt):
    """Run a request to Claude with tool use capabilities"""
    console.print(f"\n[bold blue]Sending tool-use request to Claude:[/bold blue]")
    console.print(Panel(prompt))
    
    try:
        # Define available tools
        tools = [
            {
                "name": "execute_bash",
                "description": "Execute a bash command and return the output",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute"
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        ]
        
        # Format messages
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Send request to Claude with tool use
        response = client.messages.create(
            model="claude-3-sonnet-20240307", # Updated to the latest stable model
            max_tokens=1024,
            temperature=0.3,
            messages=messages,
            tools=tools,
        )
        
        # Process the response
        content = response.content
        full_response = ""
        
        for item in content:
            if item.type == "text":
                full_response += item.text + "\n"
            elif item.type == "tool_use":
                # Extract tool information correctly
                tool_name = item.name
                tool_input = json.dumps(item.input, indent=2)
                full_response += f"\n**Tool Use**: {tool_name}\n**Input**: ```json\n{tool_input}\n```\n"
        
        # Display the response
        console.print(Panel(title="Claude Response (with Tool Use)", renderable=Markdown(full_response)))
        return full_response
        
    except Exception as e:
        console.print(f"[bold red]Error calling Claude API with tool use:[/bold red]")
        console.print(f"Error: {str(e)}")
        return None

def main():
    console.print(Panel.fit(
        "[bold green]Direct Claude API Integration[/bold green]\n"
        "This script provides direct access to the Claude API.\n"
        "Usage: ./direct_claude.py \"your prompt here\"\n"
        "Add --tools flag to enable tool use."
    ))
    
    # Parse command line arguments
    use_tools = "--tools" in sys.argv
    if use_tools and len(sys.argv) > 2:
        prompt = sys.argv[2]
    elif not use_tools and len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = input("Enter your prompt: ")
    
    # Format the prompt
    formatted_prompt = f"""
# User Request

I am Claude, an AI assistant. Please respond to this request:

{prompt}
    """
    
    # Send request to Claude
    if use_tools:
        response = run_claude_with_tool_use(formatted_prompt)
    else:
        response = run_claude_request(formatted_prompt)
    
    if response:
        console.print("[bold green]✓ Request completed successfully![/bold green]")
    else:
        console.print("[bold red]✗ Request failed.[/bold red]")

if __name__ == "__main__":
    main()