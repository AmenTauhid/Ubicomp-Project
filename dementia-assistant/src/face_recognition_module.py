"""
Face Recognition Module - Identifies known people for the Dementia Assistant.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import face_recognition


class FaceRecognizer:
    """Handles face recognition using pre-loaded known faces."""

    def __init__(
        self,
        known_faces_dir: str = "data/known_faces",
        encodings_file: str = "data/face_encodings.pkl",
        relationships_file: str = "data/relationships.json",
        tolerance: float = 0.6,
        model: str = "hog"
    ):
        """
        Initialize the face recognizer.

        Args:
            known_faces_dir: Directory containing subdirectories of face images
            encodings_file: Path to cache face encodings
            relationships_file: JSON file with name-to-relationship mapping
            tolerance: How much distance between faces to consider a match (lower = stricter)
            model: Face detection model - "hog" (faster on CPU) or "cnn" (more accurate)
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.encodings_file = Path(encodings_file)
        self.relationships_file = Path(relationships_file)
        self.tolerance = tolerance
        self.model = model

        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.relationships: Dict[str, Dict[str, str]] = {}

        self._loaded = False

    def load(self) -> bool:
        """
        Load known faces and relationships.

        Returns:
            True if loaded successfully
        """
        # Load relationships
        if not self._load_relationships():
            print("Warning: Could not load relationships file")

        # Try to load cached encodings first
        if self._load_cached_encodings():
            print(f"Loaded {len(self.known_names)} face encodings from cache")
            self._loaded = True
            return True

        # Generate encodings from images
        if self._generate_encodings():
            self._save_cached_encodings()
            print(f"Generated and cached {len(self.known_names)} face encodings")
            self._loaded = True
            return True

        print("No faces loaded - add images to data/known_faces/")
        return False

    def _load_relationships(self) -> bool:
        """Load the relationships JSON file."""
        if not self.relationships_file.exists():
            return False

        try:
            with open(self.relationships_file, 'r') as f:
                self.relationships = json.load(f)
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading relationships: {e}")
            return False

    def _load_cached_encodings(self) -> bool:
        """Load pre-computed face encodings from cache."""
        if not self.encodings_file.exists():
            return False

        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
            return True
        except (pickle.PickleError, IOError, KeyError) as e:
            print(f"Error loading cached encodings: {e}")
            return False

    def _save_cached_encodings(self) -> None:
        """Save face encodings to cache file."""
        try:
            # Ensure directory exists
            self.encodings_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_encodings,
                    'names': self.known_names
                }, f)
        except IOError as e:
            print(f"Error saving cached encodings: {e}")

    def _generate_encodings(self) -> bool:
        """Generate face encodings from images in known_faces directory."""
        if not self.known_faces_dir.exists():
            print(f"Known faces directory not found: {self.known_faces_dir}")
            return False

        self.known_encodings = []
        self.known_names = []

        # Each subdirectory is a person
        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            print(f"Processing faces for: {person_name}")

            # Process each image in the person's directory
            for image_path in person_dir.iterdir():
                if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue

                try:
                    # Load and encode face
                    image = face_recognition.load_image_file(str(image_path))
                    encodings = face_recognition.face_encodings(image)

                    if len(encodings) > 0:
                        # Use first face found in image
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(person_name)
                        print(f"  Added: {image_path.name}")
                    else:
                        print(f"  No face found in: {image_path.name}")

                except Exception as e:
                    print(f"  Error processing {image_path.name}: {e}")

        return len(self.known_encodings) > 0

    def recognize(self, rgb_frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize faces in the given frame.

        Args:
            rgb_frame: RGB image as numpy array

        Returns:
            List of dictionaries with recognition results:
            [
                {
                    "name": "john",
                    "display_name": "John",
                    "relation": "your son",
                    "confidence": 0.85,
                    "location": (top, right, bottom, left)
                }
            ]
        """
        if not self._loaded or len(self.known_encodings) == 0:
            return []

        results = []

        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare to known faces
            distances = face_recognition.face_distance(self.known_encodings, face_encoding)

            if len(distances) == 0:
                continue

            # Find best match
            best_match_idx = np.argmin(distances)
            best_distance = distances[best_match_idx]

            if best_distance <= self.tolerance:
                name = self.known_names[best_match_idx]
                confidence = 1.0 - best_distance

                # Get relationship info
                rel_info = self.relationships.get(name, {})
                display_name = rel_info.get("display_name", name.replace("_", " ").title())
                relation = rel_info.get("relation", "")

                results.append({
                    "name": name,
                    "display_name": display_name,
                    "relation": relation,
                    "confidence": round(confidence, 2),
                    "location": face_location
                })
            else:
                # Unknown face detected
                results.append({
                    "name": "unknown",
                    "display_name": "Unknown Person",
                    "relation": "",
                    "confidence": 0.0,
                    "location": face_location
                })

        return results

    def recognize_single(self, rgb_frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Recognize the most prominent/closest face in frame.

        Args:
            rgb_frame: RGB image as numpy array

        Returns:
            Recognition result dict or None if no face found
        """
        results = self.recognize(rgb_frame)

        if not results:
            return None

        # Return highest confidence match, or first unknown
        known_results = [r for r in results if r["name"] != "unknown"]

        if known_results:
            return max(known_results, key=lambda x: x["confidence"])

        return results[0]

    def get_formatted_result(self, result: Dict[str, Any]) -> str:
        """
        Format a recognition result for display.

        Args:
            result: Recognition result dictionary

        Returns:
            Human-readable string
        """
        if result["name"] == "unknown":
            return "I don't recognize this person."

        display = result["display_name"]
        relation = result["relation"]
        confidence = int(result["confidence"] * 100)

        if relation:
            return f"This is {display}, {relation}. (Confidence: {confidence}%)"
        else:
            return f"This is {display}. (Confidence: {confidence}%)"

    def refresh_encodings(self) -> bool:
        """Force regeneration of face encodings from images."""
        # Delete cache
        if self.encodings_file.exists():
            self.encodings_file.unlink()

        self.known_encodings = []
        self.known_names = []

        return self.load()


def test_face_recognition():
    """Test the face recognition module."""
    print("Testing Face Recognition Module...")

    recognizer = FaceRecognizer()

    if not recognizer.load():
        print("No faces loaded. Add images to data/known_faces/<person_name>/")
        return

    print(f"\nLoaded {len(recognizer.known_names)} face(s)")
    print(f"Known people: {set(recognizer.known_names)}")


if __name__ == "__main__":
    test_face_recognition()
