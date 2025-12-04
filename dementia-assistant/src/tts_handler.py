"""
Text-to-Speech Handler - Provides voice output for the Dementia Assistant.
Uses pyttsx3 for offline speech synthesis.
"""

import threading
import queue
from typing import Optional

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("Warning: pyttsx3 not installed. Text-to-speech will be disabled.")


class TTSHandler:
    """Handles text-to-speech output using pyttsx3."""

    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Initialize the TTS handler.

        Args:
            rate: Speech rate (words per minute). Default 150 is slower for clarity.
            volume: Volume level 0.0 to 1.0
        """
        self.rate = rate
        self.volume = volume
        self.engine: Optional[pyttsx3.Engine] = None
        self._speech_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._loaded = False

    def load(self) -> bool:
        """
        Initialize the TTS engine.

        Returns:
            True if loaded successfully
        """
        if not PYTTSX3_AVAILABLE:
            print("Error: pyttsx3 package not available")
            return False

        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)

            # Try to get available voices and set a clear one
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice if available (often clearer)
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break

            self._loaded = True
            self._running = True

            # Start worker thread for non-blocking speech
            self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self._worker_thread.start()

            print("Text-to-speech initialized successfully")
            return True

        except Exception as e:
            print(f"Error initializing TTS: {e}")
            return False

    def _speech_worker(self):
        """Worker thread that processes speech queue."""
        while self._running:
            try:
                text = self._speech_queue.get(timeout=0.5)
                if text and self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")

    def speak(self, text: str, interrupt: bool = True) -> None:
        """
        Speak the given text.

        Args:
            text: Text to speak
            interrupt: If True, clear queue and speak immediately
        """
        if not self._loaded or not text:
            return

        if interrupt:
            # Clear pending speech
            while not self._speech_queue.empty():
                try:
                    self._speech_queue.get_nowait()
                except queue.Empty:
                    break

        self._speech_queue.put(text)

    def speak_person_result(self, name: str, relation: str, confidence: float) -> None:
        """
        Speak a person recognition result in a friendly way.

        Args:
            name: Person's display name
            relation: Relationship description
            confidence: Recognition confidence (0-1)
        """
        if name.lower() == "unknown":
            self.speak("I don't recognize this person.")
        elif relation:
            self.speak(f"This is {name}, {relation}.")
        else:
            self.speak(f"This is {name}.")

    def speak_object_result(self, label: str, confidence: float) -> None:
        """
        Speak an object recognition result.

        Args:
            label: Object label
            confidence: Recognition confidence (0-1)
        """
        vowels = "aeiou"
        article = "an" if label[0].lower() in vowels else "a"
        self.speak(f"This is {article} {label}.")

    def speak_no_detection(self, detection_type: str) -> None:
        """
        Speak when nothing is detected.

        Args:
            detection_type: "person" or "object"
        """
        if detection_type == "person":
            self.speak("I don't see anyone in front of the camera.")
        else:
            self.speak("I don't recognize any objects.")

    def stop(self) -> None:
        """Stop the TTS engine and worker thread."""
        self._running = False

        # Clear queue
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break

        if self.engine:
            try:
                self.engine.stop()
            except:
                pass

    def set_rate(self, rate: int) -> None:
        """Set speech rate (words per minute)."""
        self.rate = rate
        if self.engine:
            self.engine.setProperty('rate', rate)

    def set_volume(self, volume: float) -> None:
        """Set volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        if self.engine:
            self.engine.setProperty('volume', self.volume)


def test_tts():
    """Test the TTS handler."""
    print("Testing Text-to-Speech...")

    tts = TTSHandler()
    if not tts.load():
        print("Failed to load TTS")
        return

    tts.speak("Hello! I am the Dementia Assistant.")
    tts.speak("I can help you identify people and objects.")

    import time
    time.sleep(5)

    tts.stop()
    print("TTS test complete.")


if __name__ == "__main__":
    test_tts()
