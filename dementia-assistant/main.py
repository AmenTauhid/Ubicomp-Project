#!/usr/bin/env python3
"""
Dementia Assistant - Main Application

A Raspberry Pi-based assistant for dementia patients that identifies
people and objects through voice commands.

Usage:
    python main.py
"""

import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from src.camera import Camera
from src.face_recognition_module import FaceRecognizer
from src.object_recognition import ObjectRecognizer
from src.speech_handler import SpeechHandler, Command
from src.gui import DementiaAssistantGUI, AppState
from src.tts_handler import TTSHandler
from src.activity_logger import ActivityLogger


class DementiaAssistant:
    """Main application controller."""

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the Dementia Assistant.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)

        # Components
        self.camera: Optional[Camera] = None
        self.face_recognizer: Optional[FaceRecognizer] = None
        self.object_recognizer: Optional[ObjectRecognizer] = None
        self.speech_handler: Optional[SpeechHandler] = None
        self.tts: Optional[TTSHandler] = None
        self.logger: Optional[ActivityLogger] = None
        self.gui: Optional[DementiaAssistantGUI] = None

        # State
        self._running = False
        self._processing = False
        self._current_frame = None
        self._frame_lock = threading.Lock()

        # Queue for thread-safe GUI updates
        self._gui_queue = queue.Queue()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        config_file = Path(config_path)

        if not config_file.exists():
            print(f"Warning: Config file not found at {config_path}, using defaults")
            return {}

        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
            return {}

    def initialize(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if all components initialized successfully
        """
        print("=" * 50)
        print("Dementia Assistant - Initializing")
        print("=" * 50)
        print()

        success = True

        # Initialize Camera
        print("[1/7] Initializing camera...")
        cam_config = self.config.get('camera', {})
        self.camera = Camera(
            device_index=cam_config.get('device_index', 0),
            capture_width=cam_config.get('capture_width', 1280),
            capture_height=cam_config.get('capture_height', 720),
            process_width=cam_config.get('process_width', 640),
            process_height=cam_config.get('process_height', 480)
        )

        if not self.camera.open():
            print("  ERROR: Failed to open camera")
            success = False
        else:
            print("  OK: Camera initialized")

        # Initialize Face Recognition
        print("[2/7] Initializing face recognition...")
        face_config = self.config.get('face_recognition', {})
        self.face_recognizer = FaceRecognizer(
            known_faces_dir=face_config.get('known_faces_dir', 'data/known_faces'),
            encodings_file=face_config.get('encodings_file', 'data/face_encodings.pkl'),
            relationships_file=face_config.get('relationships_file', 'data/relationships.json'),
            tolerance=face_config.get('tolerance', 0.6),
            model=face_config.get('model', 'hog')
        )

        if self.face_recognizer.load():
            print("  OK: Face recognition initialized")
        else:
            print("  WARNING: No faces loaded (add images to data/known_faces/)")

        # Initialize Object Recognition
        print("[3/7] Initializing object recognition...")
        obj_config = self.config.get('object_recognition', {})
        self.object_recognizer = ObjectRecognizer(
            model_path=obj_config.get('model', 'yolov8n.pt'),
            confidence_threshold=obj_config.get('confidence_threshold', 0.5)
        )

        if self.object_recognizer.load():
            print("  OK: Object recognition initialized")
        else:
            print("  ERROR: Failed to load object recognition model")
            success = False

        # Initialize Speech Handler
        print("[4/7] Initializing speech recognition...")
        speech_config = self.config.get('speech', {})
        self.speech_handler = SpeechHandler(
            model_path=speech_config.get('model_path', 'models/vosk-model-small-en-us-0.15'),
            sample_rate=speech_config.get('sample_rate', 16000),
            record_seconds=speech_config.get('record_seconds', 4)
        )

        if self.speech_handler.load():
            print("  OK: Speech recognition initialized")
        else:
            print("  WARNING: Speech recognition not available")
            print("  (Download Vosk model to models/ directory)")

        # Initialize Text-to-Speech
        print("[5/7] Initializing text-to-speech...")
        tts_config = self.config.get('tts', {})
        if tts_config.get('enabled', True):
            self.tts = TTSHandler(
                rate=tts_config.get('rate', 140),
                volume=tts_config.get('volume', 1.0)
            )
            if self.tts.load():
                print("  OK: Text-to-speech initialized")
            else:
                print("  WARNING: Text-to-speech not available")
                self.tts = None
        else:
            print("  SKIP: Text-to-speech disabled in config")
            self.tts = None

        # Initialize Activity Logger
        print("[6/7] Initializing activity logger...")
        log_config = self.config.get('logging', {})
        if log_config.get('enabled', True):
            self.logger = ActivityLogger(
                log_dir=log_config.get('log_dir', 'logs'),
                max_entries=log_config.get('max_entries', 1000)
            )
            if self.logger.load():
                print("  OK: Activity logger initialized")
                self.logger.log_app_event("startup", {"version": "1.0"})
            else:
                print("  WARNING: Activity logger not available")
                self.logger = None
        else:
            print("  SKIP: Activity logging disabled in config")
            self.logger = None

        # Initialize GUI
        print("[7/7] Initializing GUI...")
        gui_config = self.config.get('gui', {})
        self.gui = DementiaAssistantGUI(
            title=gui_config.get('window_title', 'Dementia Assistant'),
            width=gui_config.get('window_width', 1100),
            height=gui_config.get('window_height', 900),
            high_contrast=gui_config.get('high_contrast', True)
        )
        self.gui.create()

        # Set up callbacks
        self.gui.on_ptt_press = self._on_ptt_press
        self.gui.on_ptt_release = self._on_ptt_release
        self.gui.on_identify_person = self._on_identify_person_click
        self.gui.on_identify_object = self._on_identify_object_click
        self.gui.on_close = self._on_close

        print("  OK: GUI initialized")
        print()
        print("=" * 50)
        print("Initialization complete!")
        print("=" * 50)
        print()

        return success

    def _on_ptt_press(self) -> None:
        """Handle push-to-talk button press."""
        if self._processing:
            return

        self._processing = True
        self.gui.set_state(AppState.LISTENING)

        # Start recording in a separate thread
        thread = threading.Thread(target=self._process_voice_command)
        thread.daemon = True
        thread.start()

    def _on_ptt_release(self) -> None:
        """Handle push-to-talk button release."""
        # Recording continues until timeout
        pass

    def _on_identify_person_click(self) -> None:
        """Handle 'Who is this?' button click."""
        if self._processing:
            return

        self._processing = True
        self.gui.set_state(AppState.PROCESSING)

        # Process in background thread
        thread = threading.Thread(target=self._do_identify_person)
        thread.daemon = True
        thread.start()

    def _on_identify_object_click(self) -> None:
        """Handle 'What is this?' button click."""
        if self._processing:
            return

        self._processing = True
        self.gui.set_state(AppState.PROCESSING)

        # Process in background thread
        thread = threading.Thread(target=self._do_identify_object)
        thread.daemon = True
        thread.start()

    def _schedule_gui_update(self, func, *args):
        """Schedule a GUI update to be processed on the main thread."""
        self._gui_queue.put((func, args))

    def _process_gui_queue(self):
        """Process pending GUI updates from the queue (call from main thread)."""
        while not self._gui_queue.empty():
            try:
                func, args = self._gui_queue.get_nowait()
                func(*args)
            except queue.Empty:
                break

    def _do_identify_person(self) -> None:
        """Identify person in background thread."""
        try:
            with self._frame_lock:
                frame = self._current_frame.copy() if self._current_frame is not None else None

            if frame is None:
                self._schedule_gui_update(self.gui.set_result, "Error: No camera frame available", '#ff6666')
                return

            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Recognize face
            result = self.face_recognizer.recognize_single(rgb_frame)

            if result:
                message = self.face_recognizer.get_formatted_result(result)
                is_known = result['name'] != 'unknown'
                color = self.gui.colors['success'] if is_known else self.gui.colors['warning']
                self._schedule_gui_update(self.gui.set_result, message, color)

                # Text-to-speech
                if self.tts:
                    self.tts.speak_person_result(
                        result['display_name'],
                        result['relation'],
                        result['confidence']
                    )

                # Log the recognition
                if self.logger:
                    self.logger.log_person_recognition(
                        name=result['name'],
                        display_name=result['display_name'],
                        relation=result['relation'],
                        confidence=result['confidence'],
                        recognized=is_known
                    )
            else:
                self._schedule_gui_update(
                    self.gui.set_result,
                    "I don't see anyone in front of the camera.",
                    self.gui.colors['warning']
                )

                # TTS and logging for no detection
                if self.tts:
                    self.tts.speak_no_detection("person")
                if self.logger:
                    self.logger.log_no_detection("person")

        except Exception as e:
            print(f"Error identifying person: {e}")
            self._schedule_gui_update(self.gui.set_result, f"Error: {str(e)}", self.gui.colors['error'])

        finally:
            self._processing = False
            self._schedule_gui_update(self.gui.set_state, AppState.IDLE)

    def _do_identify_object(self) -> None:
        """Identify object in background thread."""
        try:
            with self._frame_lock:
                frame = self._current_frame.copy() if self._current_frame is not None else None

            if frame is None:
                self._schedule_gui_update(self.gui.set_result, "Error: No camera frame available", '#ff6666')
                return

            # Detect objects (prioritize center of frame)
            result = self.object_recognizer.detect_center(frame)

            if result:
                message = self.object_recognizer.get_formatted_result(result)
                self._schedule_gui_update(self.gui.set_result, message, self.gui.colors['success'])

                # Text-to-speech
                if self.tts:
                    self.tts.speak_object_result(result['label'], result['confidence'])

                # Log the recognition
                if self.logger:
                    self.logger.log_object_recognition(
                        label=result['label'],
                        confidence=result['confidence']
                    )
            else:
                # Try getting any detection
                results = self.object_recognizer.detect(frame)
                if results:
                    message = self.object_recognizer.get_all_formatted(results)
                    self._schedule_gui_update(self.gui.set_result, message, self.gui.colors['success'])

                    # TTS for first object
                    if self.tts and results:
                        self.tts.speak_object_result(results[0]['label'], results[0]['confidence'])

                    # Log all objects
                    if self.logger and results:
                        self.logger.log_object_recognition(
                            label=results[0]['label'],
                            confidence=results[0]['confidence'],
                            all_objects=[r['label'] for r in results]
                        )
                else:
                    self._schedule_gui_update(
                        self.gui.set_result,
                        "I don't recognize any objects in the frame.",
                        self.gui.colors['warning']
                    )

                    # TTS and logging for no detection
                    if self.tts:
                        self.tts.speak_no_detection("object")
                    if self.logger:
                        self.logger.log_no_detection("object")

        except Exception as e:
            print(f"Error identifying object: {e}")
            self._schedule_gui_update(self.gui.set_result, f"Error: {str(e)}", self.gui.colors['error'])

        finally:
            self._processing = False
            self._schedule_gui_update(self.gui.set_state, AppState.IDLE)

    def _process_voice_command(self) -> None:
        """Process voice command in background thread."""
        try:
            # Record and recognize speech
            command, text = self.speech_handler.record_and_recognize()

            # Update GUI state
            self._schedule_gui_update(self.gui.set_state, AppState.PROCESSING)

            print(f"Recognized: '{text}' -> {command.value}")

            # Process based on command - delegate to existing handlers
            if command == Command.IDENTIFY_PERSON:
                # Get frame and identify (reuse logic from _do_identify_person)
                with self._frame_lock:
                    frame = self._current_frame.copy() if self._current_frame is not None else None

                if frame is None:
                    self._schedule_gui_update(self.gui.set_result, "Error: No camera frame available", '#ff6666')
                else:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self.face_recognizer.recognize_single(rgb_frame)
                    if result:
                        message = self.face_recognizer.get_formatted_result(result)
                        color = '#66ff66' if result['name'] != 'unknown' else '#ffaa00'
                        self._schedule_gui_update(self.gui.set_result, message, color)
                    else:
                        self._schedule_gui_update(self.gui.set_result, "I don't see anyone in front of the camera.", '#ffaa00')

            elif command == Command.IDENTIFY_OBJECT:
                with self._frame_lock:
                    frame = self._current_frame.copy() if self._current_frame is not None else None

                if frame is None:
                    self._schedule_gui_update(self.gui.set_result, "Error: No camera frame available", '#ff6666')
                else:
                    result = self.object_recognizer.detect_center(frame)
                    if result:
                        message = self.object_recognizer.get_formatted_result(result)
                        self._schedule_gui_update(self.gui.set_result, message, '#66ff66')
                    else:
                        results = self.object_recognizer.detect(frame)
                        if results:
                            message = self.object_recognizer.get_all_formatted(results)
                            self._schedule_gui_update(self.gui.set_result, message, '#66ff66')
                        else:
                            self._schedule_gui_update(self.gui.set_result, "I don't recognize any objects in the frame.", '#ffaa00')

            elif command == Command.UNKNOWN:
                self._schedule_gui_update(
                    self.gui.set_result,
                    f"I didn't understand that. Try 'Who is this?' or 'What is this?'\n(Heard: '{text}')",
                    '#ffaa00'
                )

            elif command == Command.NONE:
                self._schedule_gui_update(
                    self.gui.set_result,
                    "I didn't hear anything. Hold the button and speak clearly.",
                    '#ffaa00'
                )

        except Exception as e:
            print(f"Error processing voice command: {e}")
            self._schedule_gui_update(self.gui.set_result, f"Error: {str(e)}", '#ff6666')

        finally:
            self._processing = False
            self._schedule_gui_update(self.gui.set_state, AppState.IDLE)

    def _on_close(self) -> None:
        """Handle application close."""
        self._running = False

    def run(self) -> None:
        """Run the main application loop."""
        self._running = True

        print("Application running. Press the button and speak to identify.")
        print("Say 'Who is this?' for people or 'What is this?' for objects.")
        print()

        try:
            while self._running and self.gui.is_running():
                # Capture frame
                ret, frame = self.camera.read_frame()

                if ret and frame is not None:
                    # Store frame for recognition
                    with self._frame_lock:
                        # Use processing resolution for ML
                        self._current_frame = cv2.resize(
                            frame,
                            (self.camera.process_width, self.camera.process_height)
                        )

                    # Update GUI with display frame
                    self.gui.update_camera_frame(frame)

                # Process any pending GUI updates from background threads
                self._process_gui_queue()

                # Process GUI events
                self.gui.update()

                # Small delay to prevent CPU hogging
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        print("Cleaning up...")

        # Log shutdown
        if self.logger:
            self.logger.log_app_event("shutdown")

        # Stop TTS
        if self.tts:
            self.tts.stop()

        if self.camera:
            self.camera.close()

        if self.gui:
            try:
                self.gui.destroy()
            except:
                pass

        print("Goodbye!")


def main():
    """Main entry point."""
    app = DementiaAssistant()

    if not app.initialize():
        print("\nWarning: Some components failed to initialize.")
        print("The application may have limited functionality.")
        print()

    app.run()


if __name__ == "__main__":
    main()
