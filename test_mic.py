#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "sounddevice",
#   "numpy",
#   "rich",
# ]
# ///

import sounddevice as sd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import sys

console = Console()

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

def record_audio(device_index=None, duration=5.0, sample_rate=44100):
    """Record audio from the specified device for the specified duration"""
    console.print(f"[bold]Recording {duration} seconds of audio from device {device_index if device_index is not None else 'default'}...[/bold]")
    console.print("[yellow]Please speak into the microphone now...[/yellow]")
    
    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        device=device_index
    )
    
    # Wait for recording to complete
    sd.wait()
    
    # Calculate audio statistics to see if it's capturing properly
    audio_mean = np.mean(np.abs(recording))
    audio_max = np.max(np.abs(recording))
    audio_min = np.min(np.abs(recording))
    
    # Print audio statistics
    stats_table = Table(title="Audio Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Mean Amplitude", f"{audio_mean:.6f}")
    stats_table.add_row("Max Amplitude", f"{audio_max:.6f}")
    stats_table.add_row("Min Amplitude", f"{audio_min:.6f}")
    
    console.print(stats_table)
    
    # Determine if audio was detected
    # If mean amplitude is very low, probably no audio was captured
    if audio_mean < 0.01 and audio_max < 0.1:
        console.print("[bold red]WARNING: Very low audio levels detected. Microphone may not be capturing audio.[/bold red]")
    else:
        console.print("[bold green]Audio detected! Recording successful.[/bold green]")
    
    return recording

def main():
    console.print(Panel.fit(
        "[bold magenta]Microphone Test Utility[/bold magenta]\n"
        "This tool lists audio devices and tests microphone input."
    ))
    
    # List all devices
    list_devices()
    
    # Ask for device index
    console.print("\n[bold]Enter the device index for your microphone (Scarlett 2i2):[/bold]")
    console.print("[yellow]Press Enter for default input device[/yellow]")
    device_input = input("> ")
    
    device_index = None if device_input.strip() == "" else int(device_input)
    
    # Test recording
    record_audio(device_index)
    
    console.print("\n[bold green]Test completed.[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Test interrupted.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)