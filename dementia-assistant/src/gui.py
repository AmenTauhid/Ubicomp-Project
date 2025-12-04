"""
GUI Module - Tkinter-based interface for the Dementia Assistant.
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
from typing import Optional, Callable
from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageTk


class AppState(Enum):
    """Application states for UI updates."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"


class DementiaAssistantGUI:
    """Main GUI for the Dementia Assistant application."""

    def __init__(
        self,
        title: str = "Dementia Assistant",
        width: int = 900,
        height: int = 700
    ):
        """
        Initialize the GUI.

        Args:
            title: Window title
            width: Window width
            height: Window height
        """
        self.title = title
        self.width = width
        self.height = height

        # Callbacks
        self.on_ptt_press: Optional[Callable] = None
        self.on_ptt_release: Optional[Callable] = None
        self.on_identify_person: Optional[Callable] = None
        self.on_identify_object: Optional[Callable] = None
        self.on_close: Optional[Callable] = None

        # State
        self._state = AppState.IDLE
        self._running = False

        # Tkinter components
        self.root: Optional[tk.Tk] = None
        self.camera_label: Optional[tk.Label] = None
        self.result_label: Optional[tk.Label] = None
        self.status_label: Optional[tk.Label] = None
        self.ptt_button: Optional[tk.Button] = None
        self.person_button: Optional[tk.Button] = None
        self.object_button: Optional[tk.Button] = None

        # Image reference (prevent garbage collection)
        self._current_image: Optional[ImageTk.PhotoImage] = None

    def create(self) -> None:
        """Create and configure the GUI window."""
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.configure(bg='#2b2b2b')

        # Set minimum window size so buttons are always visible
        self.root.minsize(800, 700)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Configure fonts
        self._setup_fonts()

        # Create layout
        self._create_layout()

        self._running = True

    def _setup_fonts(self) -> None:
        """Configure custom fonts."""
        self.font_large = tkfont.Font(family="Helvetica", size=24, weight="bold")
        self.font_medium = tkfont.Font(family="Helvetica", size=16)
        self.font_small = tkfont.Font(family="Helvetica", size=12)

    def _create_layout(self) -> None:
        """Create the GUI layout."""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Title at top
        title_label = tk.Label(
            main_frame,
            text="Dementia Assistant",
            font=self.font_large,
            fg='white',
            bg='#2b2b2b'
        )
        title_label.pack(side=tk.TOP, pady=(0, 10))

        # Bottom controls container - pack FIRST so it stays at bottom
        bottom_frame = tk.Frame(main_frame, bg='#2b2b2b')
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Result display (in bottom frame)
        result_frame = tk.Frame(bottom_frame, bg='#3a3a3a', relief=tk.RAISED, bd=1)
        result_frame.pack(fill=tk.X, pady=(10, 5))

        self.result_label = tk.Label(
            result_frame,
            text="Hold the orange button and speak, or click a button below",
            font=self.font_medium,
            fg='#ffffff',
            bg='#3a3a3a',
            wraplength=self.width - 60,
            justify=tk.CENTER,
            pady=15
        )
        self.result_label.pack(fill=tk.X, padx=10)

        # Control frame (in bottom_frame)
        control_frame = tk.Frame(bottom_frame, bg='#2b2b2b')
        control_frame.pack(fill=tk.X, pady=5)

        # All buttons in one row
        button_row = tk.Frame(control_frame, bg='#2b2b2b')
        button_row.pack(pady=5)

        # Identify Person Button
        self.person_button = tk.Button(
            button_row,
            text="Who is this?",
            font=self.font_medium,
            bg='#4a90d9',
            fg='white',
            activebackground='#3a7bc8',
            activeforeground='white',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._on_person_click
        )
        self.person_button.pack(side=tk.LEFT, padx=10)

        # PTT Button for voice commands (center)
        self.ptt_button = tk.Button(
            button_row,
            text="Hold to Speak",
            font=self.font_medium,
            bg='#e67e22',
            fg='white',
            activebackground='#ff6b6b',
            activeforeground='white',
            relief=tk.RAISED,
            bd=3,
            padx=30,
            pady=10,
            cursor="hand2"
        )
        self.ptt_button.pack(side=tk.LEFT, padx=10)

        # Bind mouse events for PTT
        self.ptt_button.bind('<ButtonPress-1>', self._on_ptt_press)
        self.ptt_button.bind('<ButtonRelease-1>', self._on_ptt_release)

        # Identify Object Button
        self.object_button = tk.Button(
            button_row,
            text="What is this?",
            font=self.font_medium,
            bg='#5cb85c',
            fg='white',
            activebackground='#4a9a4a',
            activeforeground='white',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._on_object_click
        )
        self.object_button.pack(side=tk.LEFT, padx=10)

        # Status bar (in bottom_frame)
        status_frame = tk.Frame(bottom_frame, bg='#1a1a1a')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            font=self.font_small,
            fg='#888888',
            bg='#1a1a1a',
            anchor=tk.W,
            padx=10,
            pady=5
        )
        self.status_label.pack(fill=tk.X)

        # Camera feed frame - takes remaining space in center
        camera_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=2)
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.camera_label = tk.Label(camera_frame, bg='#1a1a1a')
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _on_ptt_press(self, event) -> None:
        """Handle PTT button press."""
        if self.ptt_button:
            self.ptt_button.configure(bg='#ff6b6b', text="Listening...")
        if self.on_ptt_press:
            self.on_ptt_press()

    def _on_ptt_release(self, event) -> None:
        """Handle PTT button release."""
        if self.ptt_button:
            self.ptt_button.configure(bg='#4a90d9', text="Hold to Speak")
        if self.on_ptt_release:
            self.on_ptt_release()

    def _on_person_click(self) -> None:
        """Handle 'Who is this?' button click."""
        if self.on_identify_person:
            self.on_identify_person()

    def _on_object_click(self) -> None:
        """Handle 'What is this?' button click."""
        if self.on_identify_object:
            self.on_identify_object()

    def _on_closing(self) -> None:
        """Handle window close event."""
        self._running = False
        if self.on_close:
            self.on_close()
        self.root.destroy()

    def update_camera_frame(self, frame: np.ndarray) -> None:
        """
        Update the camera display with a new frame.

        Args:
            frame: BGR image from OpenCV
        """
        if not self._running or self.camera_label is None:
            return

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get label dimensions
            label_width = self.camera_label.winfo_width()
            label_height = self.camera_label.winfo_height()

            if label_width > 1 and label_height > 1:
                # Resize to fit label while maintaining aspect ratio
                frame_h, frame_w = rgb_frame.shape[:2]
                scale = min(label_width / frame_w, label_height / frame_h)
                new_w = int(frame_w * scale)
                new_h = int(frame_h * scale)

                rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(rgb_frame)
            self._current_image = ImageTk.PhotoImage(image=pil_image)

            self.camera_label.configure(image=self._current_image)

        except Exception as e:
            print(f"Error updating camera frame: {e}")

    def set_result(self, text: str, color: str = '#ffffff') -> None:
        """
        Update the result display text.

        Args:
            text: Result text to display
            color: Text color (hex)
        """
        if self.result_label:
            self.result_label.configure(text=text, fg=color)

    def set_status(self, text: str) -> None:
        """
        Update the status bar text.

        Args:
            text: Status text to display
        """
        if self.status_label:
            self.status_label.configure(text=text)

    def set_state(self, state: AppState) -> None:
        """
        Update the application state and UI accordingly.

        Args:
            state: New application state
        """
        self._state = state

        if state == AppState.IDLE:
            self.set_status("Ready")
            if self.ptt_button:
                self.ptt_button.configure(state=tk.NORMAL)
            if self.person_button:
                self.person_button.configure(state=tk.NORMAL)
            if self.object_button:
                self.object_button.configure(state=tk.NORMAL)

        elif state == AppState.LISTENING:
            self.set_status("Listening...")
            self.set_result("Listening... speak now", '#ffcc00')

        elif state == AppState.PROCESSING:
            self.set_status("Processing...")
            self.set_result("Processing...", '#88aaff')
            if self.ptt_button:
                self.ptt_button.configure(state=tk.DISABLED)
            if self.person_button:
                self.person_button.configure(state=tk.DISABLED)
            if self.object_button:
                self.object_button.configure(state=tk.DISABLED)

    def update(self) -> None:
        """Process pending GUI events."""
        if self._running and self.root:
            self.root.update_idletasks()
            self.root.update()

    def is_running(self) -> bool:
        """Check if the GUI is still running."""
        return self._running

    def mainloop(self) -> None:
        """Start the Tkinter main loop."""
        if self.root:
            self.root.mainloop()

    def destroy(self) -> None:
        """Destroy the GUI window."""
        self._running = False
        if self.root:
            self.root.destroy()


def test_gui():
    """Test the GUI module with dummy camera feed."""
    import time

    gui = DementiaAssistantGUI()
    gui.create()

    # Dummy callbacks
    gui.on_ptt_press = lambda: print("PTT pressed")
    gui.on_ptt_release = lambda: print("PTT released")

    # Create a dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[:] = (50, 50, 50)  # Dark gray

    # Add some text
    cv2.putText(
        dummy_frame,
        "Camera Feed",
        (200, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        2
    )

    frame_count = 0

    while gui.is_running():
        # Update dummy frame
        frame_count += 1
        display_frame = dummy_frame.copy()
        cv2.putText(
            display_frame,
            f"Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        gui.update_camera_frame(display_frame)
        gui.update()

        time.sleep(0.033)  # ~30 FPS

    print("GUI closed")


if __name__ == "__main__":
    test_gui()
