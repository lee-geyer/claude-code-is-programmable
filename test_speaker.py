#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "sounddevice",
#   "numpy",
#   "rich",
#   "openai",
#   "python-dotenv",
#   "soundfile",
# ]
# ///

import sounddevice as sd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
import os
import tempfile
import sys
import openai
from openai import OpenAI

console = Console()

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    console.print("[bold red]Error: OPENAI_API_KEY not found in environment variables[/bold red]")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def list_devices():
    """List all audio devices with their details"""
    devices = sd.query_devices()
    
    table = Table(title="Audio Devices")
    table.add_column("Index", justify="right", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Inputs", justify="center")
    table.add_column("Outputs", justify="center")
    table.add_column("Default", justify="center", style="yellow")
    
    for i, device in enumerate(devices):
        inputs = str(device.get('max_input_channels', 0))
        outputs = str(device.get('max_output_channels', 0))
        default_marks = []
        
        if device.get('name') == sd.query_devices(kind='input')['name']:
            default_marks.append("Input")
        if device.get('name') == sd.query_devices(kind='output')['name']:
            default_marks.append("Output")
            
        default = ", ".join(default_marks) if default_marks else ""
        
        table.add_row(
            str(i),
            device.get('name', 'Unknown'),
            inputs,
            outputs,
            default
        )
    
    console.print(table)

def test_speaker(device_index=None, voice="nova"):
    """Test speaker output using OpenAI TTS"""
    console.print(f"[bold]Testing speaker output on device {device_index if device_index is not None else 'default'}...[/bold]")
    
    # Generate TTS audio
    test_text = "This is a test of the audio output to the speaker. If you can hear this message, your speaker is working correctly."
    
    console.print("[bold yellow]Generating TTS audio...[/bold yellow]")
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=test_text,
            speed=1.0,
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            response.stream_to_file(temp_filename)
        
        console.print(f"[bold green]Generated TTS audio file: {temp_filename}[/bold green]")
        
        # Play the audio
        console.print(f"[bold yellow]Playing audio on device {device_index}...[/bold yellow]")
        
        import soundfile as sf
        data, samplerate = sf.read(temp_filename)
        
        sd.play(data, samplerate, device=device_index)
        sd.wait()  # Wait until audio is finished playing
        
        # Clean up
        os.unlink(temp_filename)
        console.print("[bold green]Audio playback completed![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error during audio playback:[/bold red] {str(e)}")
        return False
    
    return True

def main():
    console.print(Panel.fit(
        "[bold magenta]Speaker Test Utility[/bold magenta]\n"
        "This tool lists audio devices and tests speaker output using OpenAI TTS."
    ))
    
    # List all devices
    list_devices()
    
    # Ask for device index
    console.print("\n[bold]Enter the device index for your speakers (Razer Leviathan V2):[/bold]")
    console.print("[yellow]Press Enter for default output device[/yellow]")
    device_input = input("> ")
    
    device_index = None if device_input.strip() == "" else int(device_input)
    
    # Ask for voice
    console.print("\n[bold]Choose a voice for the test:[/bold]")
    console.print("1. Nova (default)")
    console.print("2. Alloy")
    console.print("3. Echo")
    console.print("4. Fable")
    console.print("5. Onyx")
    console.print("6. Shimmer")
    voice_input = input("> ")
    
    voice_map = {
        "1": "nova",
        "2": "alloy",
        "3": "echo",
        "4": "fable",
        "5": "onyx",
        "6": "shimmer"
    }
    
    voice = voice_map.get(voice_input.strip(), "nova")
    
    # Test speaker
    success = test_speaker(device_index, voice)
    
    if success:
        console.print("\n[bold green]Test completed successfully![/bold green]")
    else:
        console.print("\n[bold red]Test failed![/bold red]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Test interrupted.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)