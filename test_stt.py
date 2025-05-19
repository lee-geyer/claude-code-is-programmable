#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "RealtimeSTT",
#   "rich",
#   "sounddevice",
# ]
# ///

import sys
import logging
from rich.console import Console
from rich.panel import Panel
import sounddevice as sd
from RealtimeSTT import AudioToTextRecorder

console = Console()

# Suppress RealtimeSTT logs
logging.basicConfig(level=logging.ERROR)
logging.getLogger("RealtimeSTT").setLevel(logging.ERROR)
logging.getLogger("transcribe").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.ERROR)
logging.getLogger("audio_recorder").setLevel(logging.ERROR)

def list_devices():
    """List all audio devices with their details"""
    devices = sd.query_devices()
    
    console.print("[bold cyan]Audio Devices:[/bold cyan]")
    
    for i, device in enumerate(devices):
        inputs = device.get('max_input_channels', 0)
        outputs = device.get('max_output_channels', 0)
        
        # Only show devices with inputs
        if inputs > 0:
            default = ""
            if device.get('name') == sd.query_devices(kind='input')['name']:
                default = " [yellow](Default Input)[/yellow]"
            
            console.print(f"[bold]{i}[/bold]: {device.get('name', 'Unknown')} - {inputs} inputs{default}")

def test_stt(device_index=None, model="small.en"):
    """Test speech-to-text using RealtimeSTT"""
    console.print(f"\n[bold magenta]Testing speech-to-text with device {device_index if device_index is not None else 'default'}...[/bold magenta]")
    console.print(f"Using model: {model}")
    
    try:
        # Set up the recorder
        recorder = AudioToTextRecorder(
            model=model,
            language="en",
            input_device_index=device_index,
            compute_type="float32",
            post_speech_silence_duration=0.8,
            initial_prompt=None,
            print_transcription_time=True,
            enable_realtime_transcription=True
        )
        
        console.print("\n[bold yellow]Speak now...[/bold yellow]")
        console.print("(Press Ctrl+C to stop)")
        
        def realtime_update(text):
            console.print(f"[dim cyan]Realtime: {text}[/dim cyan]", end="\r")
        
        # Set up the callback for realtime updates
        recorder.on_realtime_transcription_update = realtime_update
        
        # Set up the result container
        result = {"text": "", "done": False}
        
        def transcription_callback(text):
            if text:
                console.print("\n")  # Print newline first
                console.print(Panel(text, title="Transcription"))  # Then print panel
                result["text"] = text
            result["done"] = True
        
        # Get text with callback
        recorder.text(transcription_callback)
        
        # Wait for result with a simple polling loop
        import time
        timeout = 120  # seconds - increased to 2 minutes
        start_time = time.time()
        
        # Wait for the result, with periodic checks
        while not result["done"] and time.time() - start_time < timeout:
            time.sleep(0.1)  # Short sleep to prevent CPU spinning
        
        if not result["done"]:
            console.print("[bold red]Timeout waiting for speech[/bold red]")
            recorder.shutdown()
            return False
        
        # Clean up
        recorder.shutdown()
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]Error during STT test:[/bold red] {str(e)}")
        return False

def main():
    console.print(Panel.fit(
        "[bold magenta]Speech-to-Text Test Utility[/bold magenta]\n"
        "This tool tests RealtimeSTT with your microphone."
    ))
    
    # List all devices
    list_devices()
    
    # Ask for device index
    console.print("\n[bold]Enter the device index for your microphone (Scarlett 2i2):[/bold]")
    console.print("[yellow]Press Enter for default input device[/yellow]")
    device_input = input("> ")
    
    device_index = None if device_input.strip() == "" else int(device_input)
    
    # Ask for model
    console.print("\n[bold]Choose a model for STT:[/bold]")
    console.print("1. tiny.en (fastest, least accurate)")
    console.print("2. base.en (fast, decent accuracy)")
    console.print("3. small.en (recommended)")
    console.print("4. medium.en (slow, more accurate)")
    console.print("5. large-v2 (slowest, most accurate)")
    model_input = input("> ")
    
    model_map = {
        "1": "tiny.en",
        "2": "base.en",
        "3": "small.en",
        "4": "medium.en",
        "5": "large-v2"
    }
    
    model = model_map.get(model_input.strip(), "small.en")
    
    # Test STT
    success = test_stt(device_index, model)
    
    if success:
        console.print("\n[bold green]Test completed![/bold green]")
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