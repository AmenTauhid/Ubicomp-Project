"""
Object Recognition Module - Identifies common objects for the Dementia Assistant.
Uses YOLOv8 nano model for efficient detection on Raspberry Pi.
"""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Object detection will be disabled.")


class ObjectRecognizer:
    """Handles object detection using YOLOv8."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the object recognizer.

        Args:
            model_path: Path to YOLO model (will download if not present)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model: Optional[Any] = None
        self._loaded = False

    def load(self) -> bool:
        """
        Load the YOLO model.

        Returns:
            True if loaded successfully
        """
        if not YOLO_AVAILABLE:
            print("Error: ultralytics package not available")
            return False

        try:
            print(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self._loaded = True
            print("YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the given frame.

        Args:
            frame: BGR image as numpy array (OpenCV format)

        Returns:
            List of dictionaries with detection results:
            [
                {
                    "label": "cup",
                    "confidence": 0.87,
                    "box": [x1, y1, x2, y2]
                }
            ]
        """
        if not self._loaded or self.model is None:
            return []

        results = []

        try:
            # Run inference
            predictions = self.model(frame, verbose=False)

            for pred in predictions:
                boxes = pred.boxes

                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    confidence = float(boxes.conf[i])

                    if confidence < self.confidence_threshold:
                        continue

                    # Get class name
                    class_id = int(boxes.cls[i])
                    label = self.model.names[class_id]

                    # Get bounding box
                    box = boxes.xyxy[i].tolist()

                    results.append({
                        "label": label,
                        "confidence": round(confidence, 2),
                        "box": [int(b) for b in box]
                    })

        except Exception as e:
            print(f"Error during object detection: {e}")

        return results

    def detect_primary(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect the most prominent object in frame (highest confidence).

        Args:
            frame: BGR image as numpy array

        Returns:
            Detection result dict or None if no object found
        """
        detections = self.detect(frame)

        if not detections:
            return None

        # Return highest confidence detection
        return max(detections, key=lambda x: x["confidence"])

    def detect_center(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect the object closest to the center of the frame.
        Useful for "what is THIS object" when user is pointing at something.

        Args:
            frame: BGR image as numpy array

        Returns:
            Detection result dict or None if no object found
        """
        detections = self.detect(frame)

        if not detections:
            return None

        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width / 2
        center_y = frame_height / 2

        def distance_to_center(det):
            box = det["box"]
            obj_center_x = (box[0] + box[2]) / 2
            obj_center_y = (box[1] + box[3]) / 2
            return ((obj_center_x - center_x) ** 2 + (obj_center_y - center_y) ** 2) ** 0.5

        return min(detections, key=distance_to_center)

    def get_formatted_result(self, result: Dict[str, Any]) -> str:
        """
        Format a detection result for display.

        Args:
            result: Detection result dictionary

        Returns:
            Human-readable string
        """
        label = result["label"]
        confidence = int(result["confidence"] * 100)

        # Add article
        vowels = "aeiou"
        article = "an" if label[0].lower() in vowels else "a"

        return f"This is {article} {label}. (Confidence: {confidence}%)"

    def get_all_formatted(self, results: List[Dict[str, Any]], max_items: int = 5) -> str:
        """
        Format multiple detection results.

        Args:
            results: List of detection results
            max_items: Maximum number of items to list

        Returns:
            Human-readable string listing detected objects
        """
        if not results:
            return "I don't see any recognizable objects."

        # Get unique labels with highest confidence for each
        unique_objects = {}
        for r in results:
            label = r["label"]
            if label not in unique_objects or r["confidence"] > unique_objects[label]:
                unique_objects[label] = r["confidence"]

        # Sort by confidence
        sorted_objects = sorted(
            unique_objects.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_items]

        if len(sorted_objects) == 1:
            label, conf = sorted_objects[0]
            return self.get_formatted_result({"label": label, "confidence": conf})

        # Multiple objects
        items = [f"{label} ({int(conf*100)}%)" for label, conf in sorted_objects]
        return f"I can see: {', '.join(items)}"


# COCO dataset class names for reference
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def test_object_recognition():
    """Test the object recognition module."""
    print("Testing Object Recognition Module...")

    recognizer = ObjectRecognizer()

    if not recognizer.load():
        print("Failed to load YOLO model")
        return

    print("Model loaded successfully!")
    print(f"Can detect {len(COCO_CLASSES)} object types")


if __name__ == "__main__":
    test_object_recognition()
