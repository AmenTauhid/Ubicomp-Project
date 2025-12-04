"""
Camera Module - Handles USB camera capture for the Dementia Assistant.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class Camera:
    """Manages USB camera capture and frame processing."""

    def __init__(
        self,
        device_index: int = 0,
        capture_width: int = 1280,
        capture_height: int = 720,
        process_width: int = 640,
        process_height: int = 480
    ):
        """
        Initialize the camera.

        Args:
            device_index: Camera device index (0 for first USB camera)
            capture_width: Native capture width
            capture_height: Native capture height
            process_width: Width for processing (downscaled for performance)
            process_height: Height for processing
        """
        self.device_index = device_index
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.process_width = process_width
        self.process_height = process_height

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False

    def open(self) -> bool:
        """
        Open the camera device.

        Returns:
            True if camera opened successfully, False otherwise
        """
        if self._is_opened:
            return True

        self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {self.device_index}")
            return False

        # Set capture resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

        # Verify settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera opened: {int(actual_width)}x{int(actual_height)}")

        self._is_opened = True
        return True

    def close(self) -> None:
        """Release the camera device."""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False
            self.cap = None

    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        return self._is_opened and self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            Tuple of (success, frame) where frame is the BGR image or None
        """
        if not self.is_opened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        return True, frame

    def read_frame_for_display(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame suitable for GUI display (full resolution).

        Returns:
            Tuple of (success, frame)
        """
        return self.read_frame()

    def read_frame_for_processing(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame resized for ML processing (lower resolution for speed).

        Returns:
            Tuple of (success, frame) where frame is downscaled
        """
        ret, frame = self.read_frame()
        if not ret or frame is None:
            return False, None

        # Resize for processing
        processed = cv2.resize(
            frame,
            (self.process_width, self.process_height),
            interpolation=cv2.INTER_AREA
        )

        return True, processed

    def get_current_frame_rgb(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get current frame in RGB format (for face_recognition library).

        Returns:
            Tuple of (success, rgb_frame)
        """
        ret, frame = self.read_frame_for_processing()
        if not ret or frame is None:
            return False, None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return True, rgb_frame

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def test_camera():
    """Simple test to verify camera is working."""
    print("Testing camera...")

    with Camera() as cam:
        if not cam.is_opened():
            print("Failed to open camera")
            return

        print("Camera opened successfully. Press 'q' to quit.")

        while True:
            ret, frame = cam.read_frame()
            if not ret:
                print("Failed to read frame")
                break

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("Camera test complete.")


if __name__ == "__main__":
    test_camera()
