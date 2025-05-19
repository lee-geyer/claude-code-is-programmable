#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "RealtimeSTT",
#   "openai",
#   "python-dotenv",
#   "rich",
#   "numpy",
#   "sounddevice",
#   "soundfile",
#   "markdown",
#   "anthropic",
# ]
# ///

"""
# Voice to Claude API

A voice-enabled Claude assistant that allows you to interact with the Claude API using voice commands.
This tool combines RealtimeSTT for speech recognition and OpenAI TTS for speech output.

## Features
- Real-time speech recognition using RealtimeSTT
- Claude API integration for AI assistance
- Text-to-speech responses using OpenAI TTS
- Conversation history tracking
- Voice trigger activation

## Requirements
- OpenAI API key (for TTS)
- Anthropic API key (for Claude)
- Python 3.9+
- UV package manager (for dependency management)

## Usage
Run the script:
```bash
./voice_to_claude_api.py
```

Speak to the assistant using the trigger word "Athena" in your query.
For example: "Hey Athena, what's the weather like today?"

Press Ctrl+C to exit.
"""

import os
import sys
import json
import yaml
import uuid
import asyncio
import tempfile
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
import argparse
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.logging import RichHandler
from rich.syntax import Syntax
from dotenv import load_dotenv
import openai
from openai import OpenAI
from RealtimeSTT import AudioToTextRecorder
import logging
import anthropic

# Configuration - default values
TRIGGER_WORDS = ["athena", "athina", "athene", "atina", "hey athena", "hey a thing", "a theme"]  # List of possible trigger variations
STT_MODEL = "small.en"  # Options: tiny.en, base.en, small.en, medium.en, large-v2
TTS_VOICE = "nova"  # Options: alloy, echo, fable, onyx, nova, shimmer

# Audio device configuration - using confirmed working devices
INPUT_DEVICE = "Scarlett 2i2 4th Gen"  # Your audio interface microphone
OUTPUT_DEVICE = "Razer Leviathan V2"  # Your speaker system
INPUT_DEVICE_INDEX = 1  # Set this to the index you confirmed working
OUTPUT_DEVICE_INDEX = 3  # Set this to the index you confirmed working

# Claude model configuration
CLAUDE_MODEL = "claude-3-sonnet-20240229"  # Latest stable model

# Prompt templates
COMPRESS_PROMPT = """
You are an assistant that makes long technical responses more concise for voice output.
Your task is to rephrase the following text to be shorter and more conversational,
while preserving all key information. Focus only on the most important details.
Be brief but clear, as this will be spoken aloud.

IMPORTANT HANDLING FOR CODE BLOCKS:
- Do not include full code blocks in your response
- Instead, briefly mention "I've created code for X" or "Here's a script that does Y"
- For large code blocks, just say something like "I've written a Python function that handles user authentication"
- DO NOT attempt to read out the actual code syntax
- Only describe what the code does in 1 sentences maximum

Original text:
{text}

Return only the compressed text, without any explanation or introduction.
"""

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("claude_voice_assistant")

# Suppress RealtimeSTT logs and all related loggers
logging.getLogger("RealtimeSTT").setLevel(logging.ERROR)
logging.getLogger("transcribe").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.ERROR)
logging.getLogger("audio_recorder").setLevel(logging.ERROR)
logging.getLogger("whisper").setLevel(logging.ERROR)
logging.getLogger("faster_whisper.transcribe").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("openai.http_client").setLevel(logging.ERROR)
logging.getLogger("openai._client").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.ERROR)

console = Console()

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    console.print(
        f"[bold red]Error: Missing required environment variables: {', '.join(missing_vars)}[/bold red]"
    )
    console.print("Please set these in your .env file or as environment variables.")
    sys.exit(1)

# Initialize OpenAI client for TTS
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Anthropic client for Claude
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

class ClaudeVoiceAssistant:
    def __init__(
        self,
        conversation_id: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ):
        log.info("Initializing Claude Voice Assistant")
        self.recorder = None
        self.initial_prompt = initial_prompt

        # Set up conversation ID and history
        if conversation_id:
            # Use the provided ID
            self.conversation_id = conversation_id
        else:
            # Generate a short 5-character ID
            self.conversation_id = "".join(str(uuid.uuid4()).split("-")[0][:5])
        log.info(f"Using conversation ID: {self.conversation_id}")

        # Ensure output directory exists
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Set up the conversation file path
        self.conversation_file = self.output_dir / f"{self.conversation_id}.yml"

        # Load existing conversation or start a new one
        self.conversation_history = self.load_conversation_history()

        # Set up recorder
        self.setup_recorder()

    def load_conversation_history(self) -> List[Dict[str, str]]:
        """Load conversation history from YAML file if it exists"""
        if self.conversation_file.exists():
            try:
                log.info(f"Loading existing conversation from {self.conversation_file}")
                with open(self.conversation_file, "r") as f:
                    history = yaml.safe_load(f)
                    if history is None:
                        log.info("Empty conversation file, starting new conversation")
                        return []
                    log.info(f"Loaded {len(history)} conversation turns")
                    return history
            except Exception as e:
                log.error(f"Error loading conversation history: {e}")
                log.info("Starting with empty conversation history")
                return []
        else:
            log.info(
                f"No existing conversation found at {self.conversation_file}, starting new conversation"
            )
            return []

    def save_conversation_history(self) -> None:
        """Save conversation history to YAML file"""
        try:
            log.info(f"Saving conversation history to {self.conversation_file}")
            with open(self.conversation_file, "w") as f:
                yaml.dump(self.conversation_history, f, default_flow_style=False)
            log.info(f"Saved {len(self.conversation_history)} conversation turns")
        except Exception as e:
            log.error(f"Error saving conversation history: {e}")
            console.print(
                f"[bold red]Failed to save conversation history: {e}[/bold red]"
            )

    def setup_recorder(self):
        """Set up the RealtimeSTT recorder"""
        log.info(f"Setting up STT recorder with model {STT_MODEL}")

        # Use the pre-configured input device index that we've confirmed works
        input_device_index = INPUT_DEVICE_INDEX
        log.info(f"Using configured input device index: {input_device_index}")
        
        # Print some additional info about the device
        try:
            devices = sd.query_devices()
            device_info = devices[input_device_index]
            log.info(f"Input device name: {device_info['name']}")
            log.info(f"Input device channels: {device_info['max_input_channels']}")
            log.info(f"Input device sample rate: {device_info['default_samplerate']}")
        except Exception as e:
            log.warning(f"Could not get detailed device info: {e}")

        # Create the recorder with our confirmed working device
        self.recorder = AudioToTextRecorder(
            model=STT_MODEL,
            language="en",
            compute_type="float32",
            post_speech_silence_duration=1.2,  # Slightly longer pause to better detect end of speech
            beam_size=5,
            initial_prompt=None,
            spinner=False,
            print_transcription_time=False,
            enable_realtime_transcription=True,
            realtime_model_type="tiny.en",
            realtime_processing_pause=0.4,
            input_device_index=input_device_index,
        )

        log.info(f"STT recorder initialized with model {STT_MODEL}")

    def format_messages_for_claude(self) -> List[Dict]:
        """Format the conversation history for Claude API"""
        if not self.conversation_history:
            return []

        # Convert our YAML format to Claude's message format
        claude_messages = []
        
        for entry in self.conversation_history:
            role = entry["role"]
            content = entry["content"]
            
            # Map our roles to Claude's expected roles
            if role == "user":
                claude_role = "user"
            elif role == "assistant":
                claude_role = "assistant"
            else:
                continue  # Skip unknown roles
                
            claude_messages.append({
                "role": claude_role,
                "content": content
            })
            
        return claude_messages

    async def listen(self) -> str:
        """Listen for user speech and convert to text"""
        log.info("Listening for speech...")

        # If this is the first call and we have an initial prompt, use it instead of recording
        if hasattr(self, "initial_prompt") and self.initial_prompt:
            prompt = self.initial_prompt

            # Display the prompt as if it were spoken
            console.print(
                Panel(title="You", title_align="left", renderable=Markdown(prompt))
            )

            # Clear the initial prompt so it's only used once
            self.initial_prompt = None

            return prompt

        # Create a future to signal completion
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        result_text = ""

        # Set up realtime display with improved feedback
        def on_realtime_update(text):
            # Clear line and update realtime text with some visual indicator
            sys.stdout.write("\r\033[K")  # Clear line
            sys.stdout.write(f"Listening: [realtime] {text}")
            sys.stdout.flush()

        # Callback for when transcription is complete
        def callback(text):
            nonlocal result_text
            
            # Clear the realtime display line regardless of result
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            
            if text and text.strip():  # Check if text exists and is not just whitespace
                console.print("")
                console.print(
                    Panel(title="You", title_align="left", renderable=Markdown(text))
                )
                log.info(f'Heard: "{text}"')
                result_text = text
                
                # Set the future with the result
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, text)
            else:
                log.warning("No speech detected or transcription failed")
                console.print("[yellow]No transcription produced. Try speaking louder or closer to the microphone.[/yellow]")
                
                # Set the future with empty string
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, "")

        # Callbacks for recording status changes
        def on_recording_start():
            log.info("Recording started")
            console.print("[cyan]Recording started... [/cyan]")
            
        def on_recording_stop():
            log.info("Recording stopped, processing transcription")
            console.print("[cyan]Processing speech...[/cyan]")
            
        # Set all the callbacks
        self.recorder.on_realtime_transcription_update = on_realtime_update
        self.recorder.on_recording_start = on_recording_start
        self.recorder.on_recording_stop = on_recording_stop
            
        try:
            # Start the recording in a non-blocking way
            self.recorder.text(callback)
            
            # Wait for the future to be set, with a timeout
            try:
                await asyncio.wait_for(future, timeout=30)
                return future.result()
            except asyncio.TimeoutError:
                log.warning("Timeout waiting for speech")
                console.print("[bold red]Timeout waiting for speech transcription.[/bold red]")
                return ""
                
        except Exception as e:
            log.error(f"Error during speech recognition: {str(e)}", exc_info=True)
            console.print(f"[bold red]Error during speech recognition:[/bold red] {str(e)}")
            return ""

    async def compress_speech(self, text: str) -> str:
        """Compress the response text to be more concise for speech"""
        log.info("Compressing response for speech...")

        try:
            # Use the prompt template from the constants
            prompt = COMPRESS_PROMPT.format(text=text)

            # Call OpenAI with GPT-4.1-mini to compress the text
            response = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )

            compressed_text = response.choices[0].message.content
            log.info(
                f"Compressed response from {len(text)} to {len(compressed_text)} characters"
            )
            # Display in console
            console.print(
                Panel(
                    f"[bold cyan]Original response:[/bold cyan]\n{text[:200]}...",
                    title="Original Text",
                    border_style="cyan",
                )
            )
            console.print(
                Panel(
                    f"[bold green]Compressed for speech:[/bold green]\n{compressed_text}",
                    title="Compressed Text",
                    border_style="green",
                )
            )

            return compressed_text

        except Exception as e:
            log.error(f"Error compressing speech: {str(e)}")
            console.print(f"[bold red]Error compressing speech:[/bold red] {str(e)}")
            # Return original text if compression fails
            return text

    async def speak(self, text: str):
        """Convert text to speech using OpenAI TTS"""
        log.info(f'Speaking: "{text[:50]}..."')

        try:
            # Compress text before converting to speech
            compressed_text = await self.compress_speech(text)

            # Generate speech with compressed text
            response = openai_client.audio.speech.create(
                model="tts-1",
                voice=TTS_VOICE,
                input=compressed_text,
                speed=1.0,
            )

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                response.stream_to_file(temp_filename)

            # Use the pre-configured output device index that we've confirmed works
            output_device_index = OUTPUT_DEVICE_INDEX
            log.info(f"Using configured output device index: {output_device_index}")
            
            # Print some additional info about the output device
            try:
                devices = sd.query_devices()
                device_info = devices[output_device_index]
                log.info(f"Output device name: {device_info['name']}")
                log.info(f"Output device channels: {device_info['max_output_channels']}")
            except Exception as e:
                log.warning(f"Could not get detailed output device info: {e}")

            # Play audio on the specified device
            data, samplerate = sf.read(temp_filename)
            log.info(f"Playing audio ({len(data)} samples at {samplerate}Hz)")
            
            # Use the confirmed output device
            sd.play(data, samplerate, device=output_device_index)

            # Log start time for duration tracking
            start_time = asyncio.get_event_loop().time()

            # Wait for audio to finish
            sd.wait()

            # Calculate speech duration
            duration = asyncio.get_event_loop().time() - start_time

            # Clean up the temporary file
            os.unlink(temp_filename)

            log.info(f"Audio played (duration: {duration:.2f}s)")

        except Exception as e:
            log.error(f"Error in speech synthesis: {str(e)}")
            console.print(f"[bold red]Error in speech synthesis:[/bold red] {str(e)}")
            # Display the text as fallback
            console.print(f"[italic yellow]Text:[/italic yellow] {text}")

    async def process_message(self, message: str) -> Optional[str]:
        """Process the user message and run Claude API"""
        log.info(f'Processing message: "{message}"')

        # Enhanced trigger word detection with better logging 
        trigger_found = False
        message_lower = message.lower()  # Convert to lowercase once for efficiency
        
        # Debug logging
        log.info(f"All trigger words: {TRIGGER_WORDS}")
        log.info(f"Message lowercase: '{message_lower}'")
        
        # Print each word in the message for detailed debugging
        message_words = message_lower.split()
        log.info(f"Message split into words: {message_words}")
        
        # Check each trigger word against each word in the message
        for trigger in TRIGGER_WORDS:
            trigger_lower = trigger.lower()
            
            # First try exact word match (more reliable)
            if trigger_lower in message_words:
                trigger_found = True
                log.info(f"SUCCESS! Trigger word '{trigger}' found as exact word in message")
                console.print(f"[bold green]Trigger word detected: {trigger} (exact match)[/bold green]")
                break
                
            # Then try substring match
            if trigger_lower in message_lower:
                trigger_found = True
                log.info(f"SUCCESS! Trigger word '{trigger}' found as substring in message")
                console.print(f"[bold green]Trigger word detected: {trigger} (substring match)[/bold green]")
                break
                
        if not trigger_found:
            log.info(f"No trigger word detected in message")
            console.print(f"[yellow]No trigger word detected. Looking for: {', '.join(TRIGGER_WORDS)}[/yellow]")
            return None
            
        # At this point, a trigger word was found - proceed with processing

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})

        # Format messages for Claude
        claude_messages = self.format_messages_for_claude()
        
        # Debug the messages being sent
        log.info(f"Formatted {len(claude_messages)} messages for Claude")
        for i, msg in enumerate(claude_messages):
            log.info(f"Message {i+1}: role={msg['role']}, content_length={len(msg['content'])}")
        
        console.print("\n[bold blue]🔄 Calling Claude API...[/bold blue]")
        
        # Show the message we're asking about
        console.print(Panel(
            f"[bold]Asking Claude:[/bold] {message}",
            title="API Request",
            border_style="blue"
        ))

        # List of models to try in case the primary one fails
        models_to_try = [
            CLAUDE_MODEL,  # Try the configured model first
            "claude-3-sonnet-20240229",  # Try the specific version
            "claude-3-sonnet",  # Try the generic name
            "claude-3-opus-20240229",  # Try Opus if Sonnet fails
            "claude-3-haiku-20240307"   # Try Haiku as last resort
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]
        
        last_error = None
        
        # Try each model in sequence
        for model in models_to_try:
            try:
                # Call Claude API with verbose logging
                log.info(f"Sending request to Claude API with model {model}")
                console.print(f"[dim]Calling Claude API with model {model} ({len(claude_messages)} messages)...[/dim]")
                
                response = anthropic_client.messages.create(
                    model=model,
                    max_tokens=1024,
                    temperature=0.7,
                    messages=claude_messages,
                )
                
                # Extract the response text
                response_text = response.content[0].text
                
                log.info(f"Claude API responded with {len(response_text)} characters")

                # Display the response
                console.print(Panel(title=f"Claude Response ({model})", renderable=Markdown(response_text)))

                # Add to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_text})

                # Save the updated conversation history
                self.save_conversation_history()

                return response_text

            except Exception as e:
                error_msg = f"Claude API request with model {model} failed: {str(e)}"
                log.warning(error_msg)
                console.print(f"[yellow]Model {model} failed, trying next model if available...[/yellow]")
                last_error = e
                continue
        
        # If we got here, all models failed
        error_msg = f"All Claude API models failed. Last error: {str(last_error)}"
        log.error(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")

        error_response = "I'm sorry, but I encountered an error while processing your request. Please try again."
        self.conversation_history.append(
            {"role": "assistant", "content": error_response}
        )

        # Save the updated conversation history even when there's an error
        self.save_conversation_history()

        return error_response

    async def conversation_loop(self):
        """Run the main conversation loop"""
        log.info("Starting conversation loop")

        console.print(
            Panel.fit(
                "[bold magenta]🎤 Claude Voice Assistant Ready[/bold magenta]\n"
                f"Speak to interact. Include one of these trigger words to activate: {', '.join(TRIGGER_WORDS)}.\n"
                f"The assistant will listen, process with Claude API, and respond using voice '{TTS_VOICE}'.\n"
                f"STT model: {STT_MODEL}\n"
                f"Claude model: {CLAUDE_MODEL} (Sonnet)\n"
                f"Input device: {INPUT_DEVICE} (index: {INPUT_DEVICE_INDEX})\n"
                f"Output device: {OUTPUT_DEVICE} (index: {OUTPUT_DEVICE_INDEX})\n"
                f"Conversation ID: {self.conversation_id}\n"
                f"Saving conversation to: {self.conversation_file}\n"
                f"Press Ctrl+C to exit."
            )
        )

        try:
            # Keep track of consecutive failures - set to 2 for faster debugging
            consecutive_failures = 0
            max_consecutive_failures = 2
            
            while True:
                try:
                    log.info("Listening for speech...")
                    console.print("\n[bold green]Listening... say something (include 'athena' to activate)[/bold green]")
                    
                    user_text = await self.listen()

                    # Handle case where text is None or empty
                    if user_text is None or user_text.strip() == "":
                        # This should be rare with the new async implementation, but we'll keep the check
                        console.print("[yellow]No speech detected or empty transcription. Try again.[/yellow]")
                        log.warning(f"Empty or None speech transcription received: '{user_text}'")
                        consecutive_failures += 1
                        
                        if consecutive_failures >= max_consecutive_failures:
                            log.warning(f"Hit {consecutive_failures} consecutive failures. Restarting recorder...")
                            
                            # Try to restart the recorder safely
                            if hasattr(self, "recorder") and self.recorder:
                                try:
                                    log.info("Shutting down recorder before restart...")
                                    self.recorder.shutdown()
                                    log.info("Recorder shutdown completed")
                                except Exception as e:
                                    log.error(f"Error shutting down recorder: {e}")
                            
                            # Add a small delay before reinitialization
                            await asyncio.sleep(1)
                            
                            # Reinitialize the recorder
                            log.info("Reinitializing recorder...")
                            self.setup_recorder()
                            log.info("Recorder reinitialized")
                            consecutive_failures = 0
                            
                        continue
                    
                    # Reset failure counter on successful speech detection
                    consecutive_failures = 0
                    
                    # Only process if we have actual text content
                    if user_text and user_text.strip():
                        log.info(f"Successfully transcribed: \"{user_text}\"")
                        
                        # Show detailed logging for debugging trigger word detection
                        message_words = user_text.lower().split()
                        log.info(f"Words in transcribed message: {message_words}")
                        
                        # Process message through Claude
                        response = await self.process_message(user_text)

                        # Only speak if we got a response (trigger word was detected)
                        if response:
                            log.info("Trigger word detected, processing and speaking response")
                            await self.speak(response)
                            # Give a small break between interactions
                            await asyncio.sleep(0.5)
                        else:
                            # If no trigger word, just continue listening
                            log.info("No trigger word detected in: " + user_text)
                            console.print(
                                f"[yellow]No trigger word detected. Please include one of these words: {', '.join(TRIGGER_WORDS)}. Continuing to listen...[/yellow]"
                            )
                    else:
                        log.warning("Empty or whitespace-only transcription received - skipping processing")
                        console.print("[yellow]Empty transcription. Please speak again.[/yellow]")
                
                except Exception as loop_error:
                    log.error(f"Error in conversation iteration: {str(loop_error)}", exc_info=True)
                    console.print(f"[bold red]Error in conversation loop:[/bold red] {str(loop_error)}")
                    console.print("[yellow]Continuing to listen...[/yellow]")
                    
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        log.warning(f"Hit {consecutive_failures} consecutive failures. Restarting recorder...")
                        # Try to restart the recorder
                        if hasattr(self, "recorder") and self.recorder:
                            try:
                                self.recorder.shutdown()
                            except Exception:
                                pass  # Ignore errors during shutdown
                            
                        # Reinitialize the recorder
                        self.setup_recorder()
                        consecutive_failures = 0

        except KeyboardInterrupt:
            console.print("\n[bold red]Stopping assistant...[/bold red]")
            log.info("Conversation loop stopped by keyboard interrupt")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            log.error(f"Error in conversation loop: {str(e)}", exc_info=True)
        finally:
            # Safe cleanup
            try:
                if hasattr(self, "recorder") and self.recorder:
                    log.info("Shutting down recorder...")
                    # Shutdown the recorder properly
                    self.recorder.shutdown()
            except Exception as shutdown_error:
                log.error(f"Error during shutdown: {str(shutdown_error)}")

            console.print("[bold red]Assistant stopped.[/bold red]")
            log.info("Conversation loop ended")


async def main():
    """Main entry point for the assistant"""
    log.info("Starting Claude Voice Assistant")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Voice-enabled Claude assistant")
    parser.add_argument(
        "--id",
        "-i",
        type=str,
        help="Unique ID for the conversation. If provided and exists, will load existing conversation.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="Initial prompt to process immediately (will be prefixed with trigger word)",
    )
    args = parser.parse_args()

    # Create assistant instance with conversation ID and initial prompt
    assistant = ClaudeVoiceAssistant(conversation_id=args.id, initial_prompt=args.prompt)

    # Show some helpful information about the conversation
    if args.id:
        if assistant.conversation_file.exists():
            log.info(f"Resuming existing conversation with ID: {args.id}")
            console.print(
                f"[bold green]Resuming conversation {args.id} with {len(assistant.conversation_history)} turns[/bold green]"
            )
        else:
            log.info(f"Starting new conversation with user-provided ID: {args.id}")
            console.print(
                f"[bold blue]Starting new conversation with ID: {args.id}[/bold blue]"
            )
    else:
        log.info(
            f"Starting new conversation with auto-generated ID: {assistant.conversation_id}"
        )
        console.print(
            f"[bold blue]Starting new conversation with auto-generated ID: {assistant.conversation_id}[/bold blue]"
        )

    log.info(f"Conversation will be saved to: {assistant.conversation_file}")
    console.print(f"[bold]Conversation file: {assistant.conversation_file}[/bold]")

    # Process initial prompt if provided
    if args.prompt:
        log.info(f"Processing initial prompt: {args.prompt}")
        console.print(
            f"[bold cyan]Processing initial prompt: {args.prompt}[/bold cyan]"
        )

        # Create a full prompt that includes the trigger word to ensure it's processed
        initial_prompt = f"{TRIGGER_WORDS[0]} {args.prompt}"

        # Process the initial prompt
        response = await assistant.process_message(initial_prompt)

        # Speak the response if there is one
        if response:
            await assistant.speak(response)

    # Run the conversation loop
    await assistant.conversation_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Program terminated by user")
        console.print("\n[bold red]Program terminated by user.[/bold red]")