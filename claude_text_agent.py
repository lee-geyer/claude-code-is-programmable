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
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
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

# Tool definitions
TOOLS = [
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

def execute_bash(command):
    """Execute a bash command and return the result"""
    console.print(f"[bold yellow]Executing bash command:[/bold yellow] {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return {"success": True, "output": result.stdout}
        else:
            return {"success": False, "error": result.stderr}
    except Exception as e:
        return {"success": False, "error": str(e)}

def read_file(file_path):
    """Read a file and return its contents"""
    console.print(f"[bold yellow]Reading file:[/bold yellow] {file_path}")
    try:
        path = Path(file_path)
        if not path.is_file():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        with open(path, 'r') as f:
            content = f.read()
        
        return {"success": True, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_tool(tool_call):
    """Execute a tool based on the tool call"""
    tool_name = tool_call.name
    tool_input = tool_call.input
    
    if tool_name == "execute_bash":
        return execute_bash(tool_input["command"])
    elif tool_name == "read_file":
        return read_file(tool_input["file_path"])
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}

def chat_with_claude(prompt, message_history=None):
    """Chat with Claude, handling tool use"""
    if message_history is None:
        message_history = []
    
    # Add user message to history
    message_history.append({
        "role": "user",
        "content": prompt
    })
    
    console.print(f"\n[bold blue]Sending request to Claude:[/bold blue]")
    console.print(Panel(prompt))
    
    try:
        # Send request to Claude
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0.3,
            messages=message_history,
            tools=TOOLS,
        )
        
        # Check for tool use
        tool_calls = []
        for item in response.content:
            if item.type == "tool_use":
                tool_calls.append(item)
        
        # If there are tool calls, handle them
        if tool_calls:
            console.print(f"[bold cyan]Claude wants to use {len(tool_calls)} tools[/bold cyan]")
            
            # Process each tool call
            tool_results = []
            for tool_call in tool_calls:
                console.print(Panel(
                    f"Tool: [bold]{tool_call.name}[/bold]\nInput: {json.dumps(tool_call.input, indent=2)}",
                    title="Tool Call"
                ))
                
                # Execute the tool
                result = execute_tool(tool_call)
                
                # Format the result
                if result["success"]:
                    result_output = result.get("output", result.get("content", "No output"))
                    console.print(Panel(Syntax(result_output[:500], "bash"), title="Tool Result (truncated)"))
                else:
                    error_message = result.get("error", "Unknown error")
                    console.print(Panel(f"[bold red]Error:[/bold red] {error_message}", title="Tool Error"))
                
                # Add to results
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(result)
                })
            
            # Add tool results to message history
            message_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Add tool results to message history
            message_history.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_results": tool_results
                    }
                ]
            })
            
            # Get Claude's response to the tool results
            console.print("[bold blue]Getting Claude's response to tool results...[/bold blue]")
            final_response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                temperature=0.3,
                messages=message_history,
                tools=TOOLS,
            )
            
            # Display final response
            final_text = ""
            for item in final_response.content:
                if item.type == "text":
                    final_text += item.text
            
            console.print(Panel(title="Claude's Final Response", renderable=Markdown(final_text)))
            
            # Add final response to history
            message_history.append({
                "role": "assistant",
                "content": final_response.content
            })
            
            return final_text, message_history
        else:
            # If no tool use, just get the text response
            response_text = ""
            for item in response.content:
                if item.type == "text":
                    response_text += item.text
            
            console.print(Panel(title="Claude Response", renderable=Markdown(response_text)))
            
            # Add response to history
            message_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            return response_text, message_history
            
    except Exception as e:
        console.print(f"[bold red]Error calling Claude API:[/bold red]")
        console.print(f"Error: {str(e)}")
        return None, message_history

def main():
    console.print(Panel.fit(
        "[bold green]Claude Text Agent with Tool Use[/bold green]\n"
        "This script provides an interactive chat with Claude that can execute tools.\n"
        "Type 'exit' to quit the conversation."
    ))
    
    message_history = []
    
    try:
        while True:
            # Get user input
            user_input = input("\n[bold green]Your message:[/bold green] ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("[bold yellow]Exiting conversation...[/bold yellow]")
                break
            
            # Chat with Claude
            _, message_history = chat_with_claude(user_input, message_history)
            
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Conversation interrupted. Exiting...[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
    
    console.print("[bold green]Conversation ended.[/bold green]")

if __name__ == "__main__":
    main()