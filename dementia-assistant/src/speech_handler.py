"""
Speech Handler Module - Handles voice commands for the Dementia Assistant.
Uses Vosk for offline speech recognition.
"""

import os
import json
import queue
from pathlib import Path
from typing import Optional, Callable
from enum import Enum

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not installed. Speech will be disabled.")

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("Warning: vosk not installed. Speech will be disabled.")


class Command(Enum):
    """Recognized voice commands."""
    NONE = "none"
    IDENTIFY_PERSON = "identify_person"
    IDENTIFY_OBJECT = "identify_object"
    UNKNOWN = "unknown"


class SpeechHandler:
    """Handles voice command recognition using Vosk."""

    # Keywords for command detection
    PERSON_KEYWORDS = ["who", "person", "face", "man", "woman", "people"]
    OBJECT_KEYWORDS = ["what", "object", "thing", "this", "that", "item"]

    def __init__(
        self,
        model_path: str = "models/vosk-model-small-en-us-0.15",
        sample_rate: int = 16000,
        record_seconds: float = 4.0
    ):
        """
        Initialize the speech handler.

        Args:
            model_path: Path to Vosk model directory
            sample_rate: Audio sample rate in Hz
            record_seconds: How long to record for each command
        """
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds

        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None
        self._loaded = False

        # Audio queue for recording
        self._audio_queue: queue.Queue = queue.Queue()

    def load(self) -> bool:
        """
        Load the Vosk model.

        Returns:
            True if loaded successfully
        """
        if not VOSK_AVAILABLE:
            print("Error: vosk package not available")
            return False

        if not SOUNDDEVICE_AVAILABLE:
            print("Error: sounddevice package not available")
            return False

        if not self.model_path.exists():
            print(f"Error: Vosk model not found at {self.model_path}")
            print("Download it with: wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
            return False

        try:
            print(f"Loading Vosk model from: {self.model_path}")
            self.model = Model(str(self.model_path))
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self._loaded = True
            print("Vosk model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading Vosk model: {e}")
            return False

    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio recording."""
        if status:
            print(f"Audio status: {status}")
        self._audio_queue.put(bytes(indata))

    def record_and_recognize(
        self,
        on_recording_start: Optional[Callable] = None,
        on_recording_end: Optional[Callable] = None
    ) -> tuple[Command, str]:
        """
        Record audio and recognize speech.

        Args:
            on_recording_start: Callback when recording starts
            on_recording_end: Callback when recording ends

        Returns:
            Tuple of (Command enum, raw transcription text)
        """
        if not self._loaded:
            return Command.NONE, ""

        # Clear queue
        while not self._audio_queue.empty():
            self._audio_queue.get()

        # Reset recognizer
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

        if on_recording_start:
            on_recording_start()

        try:
            # Record audio
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=8000,
                dtype='int16',
                channels=1,
                callback=self._audio_callback
            ):
                import time
                start_time = time.time()

                while time.time() - start_time < self.record_seconds:
                    try:
                        data = self._audio_queue.get(timeout=0.5)
                        self.recognizer.AcceptWaveform(data)
                    except queue.Empty:
                        continue

        except Exception as e:
            print(f"Error recording audio: {e}")
            if on_recording_end:
                on_recording_end()
            return Command.NONE, ""

        if on_recording_end:
            on_recording_end()

        # Get final result
        result = json.loads(self.recognizer.FinalResult())
        text = result.get("text", "").lower().strip()

        if not text:
            return Command.NONE, ""

        # Parse command
        command = self._parse_command(text)
        return command, text

    def _parse_command(self, text: str) -> Command:
        """
        Parse transcribed text into a command.

        Args:
            text: Transcribed speech text

        Returns:
            Detected command
        """
        text_lower = text.lower()

        # Check for person-related keywords
        has_person_keyword = any(kw in text_lower for kw in self.PERSON_KEYWORDS)

        # Check for object-related keywords
        has_object_keyword = any(kw in text_lower for kw in self.OBJECT_KEYWORDS)

        # Prioritize based on question words
        if "who" in text_lower:
            return Command.IDENTIFY_PERSON
        elif "what" in text_lower:
            return Command.IDENTIFY_OBJECT
        elif has_person_keyword:
            return Command.IDENTIFY_PERSON
        elif has_object_keyword:
            return Command.IDENTIFY_OBJECT

        # If we got any speech but couldn't parse it
        if text:
            return Command.UNKNOWN

        return Command.NONE

    def get_command_description(self, command: Command) -> str:
        """Get human-readable description of a command."""
        descriptions = {
            Command.NONE: "No speech detected",
            Command.IDENTIFY_PERSON: "Identifying person...",
            Command.IDENTIFY_OBJECT: "Identifying object...",
            Command.UNKNOWN: "Command not recognized"
        }
        return descriptions.get(command, "Unknown command")

    def list_audio_devices(self) -> list:
        """List available audio input devices."""
        if not SOUNDDEVICE_AVAILABLE:
            return []

        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })

        return input_devices


def test_speech_handler():
    """Test the speech handler module."""
    print("Testing Speech Handler Module...")
    print()

    handler = SpeechHandler()

    # List audio devices
    print("Available audio input devices:")
    devices = handler.list_audio_devices()
    for d in devices:
        print(f"  [{d['index']}] {d['name']}")
    print()

    if not handler.load():
        print("Failed to load speech model")
        return

    print("Say something (recording for 4 seconds)...")
    print()

    command, text = handler.record_and_recognize(
        on_recording_start=lambda: print(">> Recording..."),
        on_recording_end=lambda: print(">> Done recording")
    )

    print(f"Transcription: '{text}'")
    print(f"Command: {command.value}")
    print(f"Description: {handler.get_command_description(command)}")


if __name__ == "__main__":
    test_speech_handler()
