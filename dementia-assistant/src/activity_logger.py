"""
Activity Logger - Logs all recognitions for caregiver review.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class ActivityLogger:
    """Logs recognition events with timestamps for caregiver review."""

    def __init__(self, log_dir: str = "logs", max_entries: int = 1000):
        """
        Initialize the activity logger.

        Args:
            log_dir: Directory to store log files
            max_entries: Maximum entries per log file before rotation
        """
        self.log_dir = Path(log_dir)
        self.max_entries = max_entries
        self._current_log: List[Dict[str, Any]] = []
        self._log_file: Optional[Path] = None
        self._loaded = False

    def load(self) -> bool:
        """
        Initialize the logger and create log directory.

        Returns:
            True if initialized successfully
        """
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = self._get_current_log_file()
            self._load_existing_log()
            self._loaded = True
            print(f"Activity logger initialized: {self.log_dir}")
            return True
        except Exception as e:
            print(f"Error initializing activity logger: {e}")
            return False

    def _get_current_log_file(self) -> Path:
        """Get the current day's log file path."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"activity_{date_str}.json"

    def _load_existing_log(self) -> None:
        """Load existing log entries for today."""
        if self._log_file and self._log_file.exists():
            try:
                with open(self._log_file, 'r') as f:
                    self._current_log = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._current_log = []
        else:
            self._current_log = []

    def _save_log(self) -> None:
        """Save current log entries to file."""
        if not self._log_file:
            return

        # Check if we need to rotate to a new day's file
        current_file = self._get_current_log_file()
        if current_file != self._log_file:
            self._log_file = current_file
            self._current_log = []

        try:
            with open(self._log_file, 'w') as f:
                json.dump(self._current_log, f, indent=2)
        except IOError as e:
            print(f"Error saving activity log: {e}")

    def log_person_recognition(
        self,
        name: str,
        display_name: str,
        relation: str,
        confidence: float,
        recognized: bool = True
    ) -> None:
        """
        Log a person recognition event.

        Args:
            name: Internal person ID
            display_name: Display name shown to user
            relation: Relationship description
            confidence: Recognition confidence
            recognized: True if person was recognized, False if unknown
        """
        if not self._loaded:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "person_recognition",
            "recognized": recognized,
            "details": {
                "name": name,
                "display_name": display_name,
                "relation": relation,
                "confidence": round(confidence, 3)
            }
        }

        self._add_entry(entry)

    def log_object_recognition(
        self,
        label: str,
        confidence: float,
        all_objects: Optional[List[str]] = None
    ) -> None:
        """
        Log an object recognition event.

        Args:
            label: Primary object detected
            confidence: Recognition confidence
            all_objects: List of all objects detected in frame
        """
        if not self._loaded:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "object_recognition",
            "details": {
                "primary_object": label,
                "confidence": round(confidence, 3),
                "all_objects": all_objects or [label]
            }
        }

        self._add_entry(entry)

    def log_no_detection(self, detection_type: str) -> None:
        """
        Log when nothing was detected.

        Args:
            detection_type: "person" or "object"
        """
        if not self._loaded:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": f"no_{detection_type}_detected",
            "details": {}
        }

        self._add_entry(entry)

    def log_voice_command(self, command: str, transcription: str) -> None:
        """
        Log a voice command.

        Args:
            command: Parsed command type
            transcription: Raw transcription text
        """
        if not self._loaded:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "voice_command",
            "details": {
                "command": command,
                "transcription": transcription
            }
        }

        self._add_entry(entry)

    def log_app_event(self, event: str, details: Optional[Dict] = None) -> None:
        """
        Log an application event (startup, shutdown, errors).

        Args:
            event: Event name
            details: Additional details
        """
        if not self._loaded:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "app_event",
            "event": event,
            "details": details or {}
        }

        self._add_entry(entry)

    def _add_entry(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the log and save."""
        self._current_log.append(entry)

        # Trim if too many entries
        if len(self._current_log) > self.max_entries:
            self._current_log = self._current_log[-self.max_entries:]

        self._save_log()

    def get_today_summary(self) -> Dict[str, Any]:
        """
        Get a summary of today's activity.

        Returns:
            Summary dictionary with counts and recent entries
        """
        if not self._loaded:
            return {}

        person_recognitions = [e for e in self._current_log if e["type"] == "person_recognition"]
        object_recognitions = [e for e in self._current_log if e["type"] == "object_recognition"]

        # Count unique people recognized
        unique_people = set()
        for e in person_recognitions:
            if e.get("recognized"):
                unique_people.add(e["details"].get("name", "unknown"))

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_events": len(self._current_log),
            "person_recognitions": len(person_recognitions),
            "object_recognitions": len(object_recognitions),
            "unique_people_seen": list(unique_people),
            "last_5_events": self._current_log[-5:] if self._current_log else []
        }

    def get_log_files(self) -> List[Path]:
        """Get list of all log files."""
        if not self.log_dir.exists():
            return []
        return sorted(self.log_dir.glob("activity_*.json"), reverse=True)


def test_activity_logger():
    """Test the activity logger."""
    print("Testing Activity Logger...")

    logger = ActivityLogger(log_dir="logs")
    if not logger.load():
        print("Failed to initialize logger")
        return

    # Log some test events
    logger.log_app_event("test_startup")
    logger.log_person_recognition("john", "John", "your son", 0.92)
    logger.log_object_recognition("cup", 0.87, ["cup", "table"])
    logger.log_no_detection("person")
    logger.log_voice_command("identify_person", "who is this")

    # Get summary
    summary = logger.get_today_summary()
    print(f"\nToday's Summary:")
    print(json.dumps(summary, indent=2))

    print("\nActivity logger test complete.")


if __name__ == "__main__":
    test_activity_logger()
