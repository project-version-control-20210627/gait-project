#!/usr/bin/env python3
"""
Gait Analysis Pose Viewer

A Tkinter-based application for viewing human pose estimation
frame-by-frame from video files. Allows evaluation of pose detection
sensitivity for gait analysis.

Usage:
    python pose_viewer.py [video_path]

If no video path is provided, a file dialog will open.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_detector import PoseDetector
from center_of_mass import CenterOfMassCalculator
from gait_cycle import (
    GaitCycleDetector, ViewTypeAnalyzer, HeelStrikeEvent, GaitCycle,
    FramePoseData, BatchProcessingResults, batch_process_video
)
from gait_normalizer import GaitNormalizer, NormalizedGaitData
from gait_exporter import GaitDataExporter
from gait_viewer_panels import (
    GaitControlPanel, HeelStrikeTimelinePanel, GaitCycleListPanel,
    CycleComparisonPanel, CenterOfMassPanel, PosturalAnglesPanel, ExportPanel
)


class VideoProcessor:
    """Handles video loading and frame extraction."""

    def __init__(self, video_path: str):
        """
        Initialize video processor.

        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get a specific frame from the video.

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            BGR frame as numpy array
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise ValueError(f"Frame index {frame_idx} out of range (0-{self.total_frames - 1})")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            raise ValueError(f"Could not read frame {frame_idx}")

        self.current_frame_idx = frame_idx
        return frame

    def get_time_str(self, frame_idx: int) -> str:
        """Get timestamp string for a frame."""
        if self.fps > 0:
            seconds = frame_idx / self.fps
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02d}:{secs:05.2f}"
        return "N/A"

    def close(self):
        """Release video resources."""
        self.cap.release()


class PoseViewerApp:
    """M


    ain Tkinter application for pose viewing."""

    def __init__(self, root: tk.Tk, video_path: str = None):
        """
        Initialize the application.

        Args:
            root: Tkinter root window
            video_path: Optional path to video file
        """
        self.root = root
        self.root.title("Gait Analysis - Pose Viewer")
        self.root.configure(bg='#2b2b2b')

        # Initialize components
        self.video_processor = None
        self.pose_detector = None
        self.current_frame = None
        self.current_landmarks = None
        self.current_frame_idx = 0
        self.display_scale = 1.0
        self.show_original = tk.BooleanVar(value=True)
        self.show_skeleton_overlay = tk.BooleanVar(value=True)

        # Gait analysis components
        self.com_calculator = CenterOfMassCalculator()
        self.gait_normalizer = GaitNormalizer()
        self.gait_exporter = GaitDataExporter()
        self.view_type = tk.StringVar(value='side')
        self.batch_results: BatchProcessingResults = None
        self.normalized_cycles: list[NormalizedGaitData] = []
        self.is_processing = False

        # Setup UI
        self._setup_ui()

        # Load video if provided
        if video_path:
            self._load_video(video_path)

        # Bind keyboard shortcuts
        self.root.bind('<Left>', lambda e: self._prev_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('<Home>', lambda e: self._goto_first_frame())
        self.root.bind('<End>', lambda e: self._goto_last_frame())
        self.root.bind('<space>', lambda e: self._toggle_overlay())

    def _setup_ui(self):
        """Create the user interface."""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=0)  # Display panels - fixed size, no expansion
        self.main_frame.rowconfigure(4, weight=1)  # Gait notebook gets remaining space

        # Style - use explicit named styles for reliable dark theme
        style = ttk.Style()
        style.configure('TButton', padding=6)

        # Dark theme styles
        style.configure('Dark.TLabel', background='#2b2b2b', foreground='white')
        style.configure('Dark.TFrame', background='#2b2b2b')
        style.configure('TLabelframe', background='#2b2b2b')
        style.configure('TLabelframe.Label', background='#2b2b2b', foreground='white')

        # High-contrast styles for angle panel (black/dark on light gray)
        light_gray_bg = '#ececec'
        style.configure('AnglePanel.TLabelframe', background=light_gray_bg)
        style.configure('AnglePanel.TLabelframe.Label', background=light_gray_bg, foreground='#000000')
        style.configure('AnglePanel.TFrame', background=light_gray_bg)
        style.configure('AngleValue.TLabel', background=light_gray_bg, foreground='#000080')  # Dark blue
        style.configure('AngleLabel.TLabel', background=light_gray_bg, foreground='#000000')  # Black
        style.configure('Header.TLabel', background=light_gray_bg, foreground='#000000')  # Black
        style.configure('Status.TLabel', background='#2b2b2b', foreground='#aaaaaa')

        # Detection status styles (on light gray background)
        style.configure('DetectGood.TLabel', background=light_gray_bg, foreground='#006600')  # Dark green
        style.configure('DetectBad.TLabel', background=light_gray_bg, foreground='#cc0000')  # Dark red

        # Apply default dark style to base classes
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TFrame', background='#2b2b2b')

        # Top control bar
        self._create_control_bar()

        # Image display area (two panels side by side)
        self._create_display_panels()

        # Bottom info panel
        self._create_info_panel()

        # Status bar
        self._create_status_bar()

        # Gait analysis panels (tabbed interface)
        self._create_gait_panels()

    def _create_control_bar(self):
        """Create the top control bar."""
        control_frame = ttk.Frame(self.main_frame, style='Dark.TFrame')
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # Load button
        self.load_btn = ttk.Button(control_frame, text="Load Video", command=self._open_file_dialog)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Navigation buttons
        nav_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        nav_frame.pack(side=tk.LEFT, padx=20)

        self.first_btn = ttk.Button(nav_frame, text="|<", width=3, command=self._goto_first_frame)
        self.first_btn.pack(side=tk.LEFT, padx=2)

        self.prev_btn = ttk.Button(nav_frame, text="<", width=3, command=self._prev_frame)
        self.prev_btn.pack(side=tk.LEFT, padx=2)

        self.next_btn = ttk.Button(nav_frame, text=">", width=3, command=self._next_frame)
        self.next_btn.pack(side=tk.LEFT, padx=2)

        self.last_btn = ttk.Button(nav_frame, text=">|", width=3, command=self._goto_last_frame)
        self.last_btn.pack(side=tk.LEFT, padx=2)

        # Frame slider
        slider_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

        self.frame_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self._on_slider_change
        )
        self.frame_slider.pack(fill=tk.X, expand=True)

        # Frame number entry
        frame_entry_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        frame_entry_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(frame_entry_frame, text="Frame:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.frame_entry = ttk.Entry(frame_entry_frame, width=8)
        self.frame_entry.pack(side=tk.LEFT, padx=5)
        self.frame_entry.bind('<Return>', self._on_frame_entry)

        self.total_frames_label = ttk.Label(frame_entry_frame, text="/ 0", style='Dark.TLabel')
        self.total_frames_label.pack(side=tk.LEFT)

        # Display options
        options_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        options_frame.pack(side=tk.RIGHT, padx=10)

        self.overlay_check = ttk.Checkbutton(
            options_frame,
            text="Show Skeleton",
            variable=self.show_skeleton_overlay,
            command=self._update_display
        )
        self.overlay_check.pack(side=tk.LEFT, padx=5)

    def _create_display_panels(self):
        """Create the image display panels."""
        # Left panel - Original with overlay
        self.left_display_frame = ttk.LabelFrame(self.main_frame, text="Original + Pose Overlay")
        self.left_display_frame.grid(row=1, column=0, sticky="nw", padx=(0, 5), pady=5)

        # Large fixed canvas for video display
        self.left_canvas = tk.Canvas(self.left_display_frame, bg='black', width=640, height=360,
                                     highlightthickness=0)
        self.left_canvas.grid(row=0, column=0, padx=5, pady=5)

        # Right panel - Stick figure only
        self.right_display_frame = ttk.LabelFrame(self.main_frame, text="Stick Figure (Isolated)")
        self.right_display_frame.grid(row=1, column=1, sticky="nw", padx=(5, 0), pady=5)

        self.right_canvas = tk.Canvas(self.right_display_frame, bg='black', width=640, height=360,
                                      highlightthickness=0)
        self.right_canvas.grid(row=0, column=0, padx=5, pady=5)

    def _create_info_panel(self):
        """Create the joint angles info panel."""
        self.info_panel_frame = ttk.LabelFrame(
            self.main_frame,
            text="Joint & Postural Angles (degrees)",
            style='AnglePanel.TLabelframe'
        )
        self.info_panel_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # Create angle labels in a grid
        self.angle_labels = {}

        # Left side angles
        left_frame = ttk.Frame(self.info_panel_frame, style='AnglePanel.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)

        ttk.Label(left_frame, text="LEFT SIDE", style='Header.TLabel',
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=2)

        angle_names_left = ['left_shoulder', 'left_elbow', 'left_hip', 'left_knee', 'left_ankle']
        for i, name in enumerate(angle_names_left):
            display_name = name.replace('left_', '').title()
            ttk.Label(left_frame, text=f"{display_name}:",
                      style='AngleLabel.TLabel').grid(row=i+1, column=0, sticky='e', padx=5)
            self.angle_labels[name] = ttk.Label(left_frame, text="--", style='AngleValue.TLabel',
                                                 font=('TkDefaultFont', 10, 'bold'))
            self.angle_labels[name].grid(row=i+1, column=1, sticky='w', padx=5)

        # Right side angles
        right_frame = ttk.Frame(self.info_panel_frame, style='AnglePanel.TFrame')
        right_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)

        ttk.Label(right_frame, text="RIGHT SIDE", style='Header.TLabel',
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=2)

        angle_names_right = ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee', 'right_ankle']
        for i, name in enumerate(angle_names_right):
            display_name = name.replace('right_', '').title()
            ttk.Label(right_frame, text=f"{display_name}:",
                      style='AngleLabel.TLabel').grid(row=i+1, column=0, sticky='e', padx=5)
            self.angle_labels[name] = ttk.Label(right_frame, text="--", style='AngleValue.TLabel',
                                                 font=('TkDefaultFont', 10, 'bold'))
            self.angle_labels[name].grid(row=i+1, column=1, sticky='w', padx=5)

        # Postural angles (view-dependent)
        postural_frame = ttk.Frame(self.info_panel_frame, style='AnglePanel.TFrame')
        postural_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)

        ttk.Label(postural_frame, text="POSTURE", style='Header.TLabel',
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=2)

        # Side view postural angles (sagittal plane)
        postural_angles_side = [
            ('cervical_flexion', 'Cervical'),
            ('thoracic_inclination', 'Thoracic'),
            ('trunk_inclination', 'Trunk Incl.'),
        ]
        # Front view postural angles (frontal plane)
        postural_angles_front = [
            ('shoulder_tilt', 'Shoulder Tilt'),
            ('hip_tilt', 'Hip Tilt'),
            ('trunk_lateral_lean', 'Lateral Lean'),
        ]

        # Create labels for all postural angles (show/hide based on view type)
        row_idx = 1
        for name, display_name in postural_angles_side:
            label = ttk.Label(postural_frame, text=f"{display_name}:",
                              style='AngleLabel.TLabel')
            label.grid(row=row_idx, column=0, sticky='e', padx=5)
            self.angle_labels[f'{name}_label'] = label

            value_label = ttk.Label(postural_frame, text="--", style='AngleValue.TLabel',
                                    font=('TkDefaultFont', 10, 'bold'))
            value_label.grid(row=row_idx, column=1, sticky='w', padx=5)
            self.angle_labels[name] = value_label
            row_idx += 1

        for name, display_name in postural_angles_front:
            label = ttk.Label(postural_frame, text=f"{display_name}:",
                              style='AngleLabel.TLabel')
            label.grid(row=row_idx, column=0, sticky='e', padx=5)
            self.angle_labels[f'{name}_label'] = label

            value_label = ttk.Label(postural_frame, text="--", style='AngleValue.TLabel',
                                    font=('TkDefaultFont', 10, 'bold'))
            value_label.grid(row=row_idx, column=1, sticky='w', padx=5)
            self.angle_labels[name] = value_label
            row_idx += 1

        # Detection confidence
        conf_frame = ttk.Frame(self.info_panel_frame, style='AnglePanel.TFrame')
        conf_frame.pack(side=tk.RIGHT, padx=20, pady=5)

        ttk.Label(conf_frame, text="Detection:", style='Header.TLabel',
                  font=('TkDefaultFont', 10, 'bold')).pack()
        self.detection_label = ttk.Label(conf_frame, text="No pose detected", style='DetectBad.TLabel')
        self.detection_label.pack()

    def _create_status_bar(self):
        """Create the status bar."""
        status_frame = ttk.Frame(self.main_frame, style='Dark.TFrame')
        status_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self.status_label = ttk.Label(status_frame, text="Load a video to begin",
                                       style='Dark.TLabel')
        self.status_label.pack(side=tk.LEFT)

        self.time_label = ttk.Label(status_frame, text="", style='AngleValue.TLabel')
        self.time_label.pack(side=tk.RIGHT)

        # Keyboard shortcuts hint
        hint_label = ttk.Label(
            status_frame,
            text="  |  Shortcuts: <- -> Navigate  |  Home/End First/Last  |  Space Toggle Skeleton",
            style='Status.TLabel'
        )
        hint_label.pack(side=tk.LEFT, padx=20)

    def _open_file_dialog(self):
        """Open a file dialog to select a video."""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("All files", "*.*")
        ]

        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes,
            initialdir=os.path.expanduser("~")
        )

        if video_path:
            self._load_video(video_path)

    def _load_video(self, video_path: str):
        """Load a video file."""
        try:
            # Close existing resources
            if self.video_processor:
                self.video_processor.close()
            if self.pose_detector:
                self.pose_detector.close()

            # Load new video
            self.video_processor = VideoProcessor(video_path)
            self.pose_detector = PoseDetector()

            # Update UI
            self.frame_slider.configure(to=self.video_processor.total_frames - 1)
            self.total_frames_label.configure(text=f"/ {self.video_processor.total_frames}")

            # Calculate display scale - match canvas size (640x360 for 16:9)
            max_display_width = 640
            max_display_height = 360
            scale_w = max_display_width / self.video_processor.width
            scale_h = max_display_height / self.video_processor.height
            self.display_scale = min(scale_w, scale_h, 1.0)

            # Update canvas sizes
            display_w = int(self.video_processor.width * self.display_scale)
            display_h = int(self.video_processor.height * self.display_scale)
            self.left_canvas.configure(width=display_w, height=display_h)
            self.right_canvas.configure(width=display_w, height=display_h)

            # Load first frame
            self._goto_frame(0)

            # Update status
            filename = os.path.basename(video_path)
            self.status_label.configure(
                text=f"Loaded: {filename} | {self.video_processor.width}x{self.video_processor.height} | "
                     f"{self.video_processor.fps:.1f} FPS | {self.video_processor.total_frames} frames"
            )

            self.root.title(f"Gait Analysis - Pose Viewer - {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video:\n{str(e)}")

    def _goto_frame(self, frame_idx: int):
        """Navigate to a specific frame."""
        if self.video_processor is None:
            return

        # Clamp frame index
        frame_idx = max(0, min(frame_idx, self.video_processor.total_frames - 1))

        try:
            # Get frame
            self.current_frame = self.video_processor.get_frame(frame_idx)
            self.current_frame_idx = frame_idx

            # Detect pose
            self.current_landmarks = self.pose_detector.detect(self.current_frame)

            # Update UI
            self.frame_slider.set(frame_idx)
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, str(frame_idx))
            self.time_label.configure(text=f"Time: {self.video_processor.get_time_str(frame_idx)}")

            # Update display
            self._update_display()

            # Update timeline if available
            self._update_timeline_frame()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load frame {frame_idx}:\n{str(e)}")

    def _update_display(self):
        """Update the display panels."""
        if self.current_frame is None:
            return

        # Left panel: Original with overlay
        if self.current_landmarks and self.show_skeleton_overlay.get():
            left_image = self.pose_detector.draw_stick_figure(
                self.current_frame,
                self.current_landmarks,
                draw_on_original=True,
                line_thickness=2,
                circle_radius=4
            )
        else:
            left_image = self.current_frame.copy()

        self._display_frame(left_image, self.left_canvas)

        # Right panel: Stick figure only
        if self.current_landmarks:
            right_image = self.pose_detector.draw_stick_figure(
                self.current_frame,
                self.current_landmarks,
                draw_on_original=False,
                line_thickness=3,
                circle_radius=6
            )
            self.detection_label.configure(text="Pose detected", style='DetectGood.TLabel')
        else:
            right_image = np.zeros_like(self.current_frame)
            cv2.putText(
                right_image,
                "No pose detected",
                (50, right_image.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            self.detection_label.configure(text="No pose detected", style='DetectBad.TLabel')

        self._display_frame(right_image, self.right_canvas)

        # Update joint angles
        self._update_angle_display()

    def _display_frame(self, frame: np.ndarray, canvas: tk.Canvas):
        """Display a frame on a canvas."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize for display
        if self.display_scale != 1.0:
            new_width = int(frame.shape[1] * self.display_scale)
            new_height = int(frame.shape[0] * self.display_scale)
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))

        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(pil_image)

        # Update canvas
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        # Keep a reference to prevent garbage collection
        canvas.photo = photo

    def _update_angle_display(self):
        """Update the joint and postural angle labels."""
        if self.current_landmarks:
            # Get all angles including postural angles (view-type dependent)
            view_type = self.view_type.get()
            angles = self.pose_detector.get_all_angles(self.current_landmarks, view_type)

            for name, label in self.angle_labels.items():
                # Skip label widgets (keys ending with '_label')
                if name.endswith('_label'):
                    continue

                angle = angles.get(name)
                if angle is not None:
                    label.configure(text=f"{angle:.1f}°")
                else:
                    label.configure(text="--")
        else:
            for name, label in self.angle_labels.items():
                # Skip label widgets
                if name.endswith('_label'):
                    continue
                label.configure(text="--")

    def _prev_frame(self):
        """Go to previous frame."""
        self._goto_frame(self.current_frame_idx - 1)

    def _next_frame(self):
        """Go to next frame."""
        self._goto_frame(self.current_frame_idx + 1)

    def _goto_first_frame(self):
        """Go to first frame."""
        self._goto_frame(0)

    def _goto_last_frame(self):
        """Go to last frame."""
        if self.video_processor:
            self._goto_frame(self.video_processor.total_frames - 1)

    def _on_slider_change(self, value):
        """Handle slider value change."""
        frame_idx = int(float(value))
        if frame_idx != self.current_frame_idx:
            self._goto_frame(frame_idx)

    def _on_frame_entry(self, event):
        """Handle frame entry submission."""
        try:
            frame_idx = int(self.frame_entry.get())
            self._goto_frame(frame_idx)
        except ValueError:
            pass

    def _toggle_overlay(self):
        """Toggle skeleton overlay."""
        self.show_skeleton_overlay.set(not self.show_skeleton_overlay.get())
        self._update_display()

    # =====================
    # Gait Analysis Methods
    # =====================

    def _create_gait_panels(self):
        """Create the gait analysis tabbed interface."""
        # Create notebook (tabbed interface)
        self.gait_notebook = ttk.Notebook(self.main_frame)
        self.gait_notebook.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(10, 0))

        # Configure row weight for notebook
        self.main_frame.rowconfigure(4, weight=1)

        # Tab 1: Gait Control Panel
        self.gait_control_panel = GaitControlPanel(
            self.gait_notebook,
            on_process=self._batch_process_video,
            on_view_change=self._on_view_type_change,
            on_auto_detect_view=self._auto_detect_view_type,
            view_type_var=self.view_type
        )
        self.gait_notebook.add(self.gait_control_panel, text="Gait Analysis")

        # Tab 2: Timeline
        self.timeline_panel = HeelStrikeTimelinePanel(
            self.gait_notebook,
            on_frame_select=self._goto_frame,
            on_event_add=self._add_heel_strike,
            on_event_remove=self._remove_heel_strike,
            on_clear_auto=self._clear_auto_detected
        )
        self.gait_notebook.add(self.timeline_panel, text="Timeline")

        # Tab 3: Cycles List
        self.cycles_panel = GaitCycleListPanel(
            self.gait_notebook,
            on_cycle_select=self._on_cycle_select
        )
        self.gait_notebook.add(self.cycles_panel, text="Cycles")

        # Tab 4: Comparison
        self.comparison_panel = CycleComparisonPanel(self.gait_notebook)
        self.gait_notebook.add(self.comparison_panel, text="Compare")

        # Tab 5: Postural Angles
        self.posture_panel = PosturalAnglesPanel(
            self.gait_notebook,
            view_type_var=self.view_type
        )
        self.gait_notebook.add(self.posture_panel, text="Posture")

        # Tab 6: Center of Mass
        self.com_panel = CenterOfMassPanel(self.gait_notebook)
        self.gait_notebook.add(self.com_panel, text="Center of Mass")

        # Tab 7: Export
        self.export_panel = ExportPanel(
            self.gait_notebook,
            on_export=self._export_data
        )
        self.gait_notebook.add(self.export_panel, text="Export")

        # Bind tab change event to show/hide video panels for graph tabs
        self.gait_notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

        # Track which tabs are "graph tabs" that should expand
        self.graph_tab_names = {'Compare', 'Posture', 'Center of Mass'}
        self.panels_hidden = False

    def _on_tab_changed(self, event):
        """Handle notebook tab change - hide/show video panels for graph tabs."""
        try:
            current_tab = self.gait_notebook.tab(self.gait_notebook.select(), 'text')
        except Exception:
            return

        if current_tab in self.graph_tab_names:
            # Hide video and angle panels to expand graph area
            if not self.panels_hidden:
                self.left_display_frame.grid_remove()
                self.right_display_frame.grid_remove()
                self.info_panel_frame.grid_remove()
                self.panels_hidden = True
                # Give more weight to the notebook row when panels are hidden
                self.main_frame.rowconfigure(1, weight=0)
                self.main_frame.rowconfigure(2, weight=0)
        else:
            # Show video and angle panels
            if self.panels_hidden:
                self.left_display_frame.grid()
                self.right_display_frame.grid()
                self.info_panel_frame.grid()
                self.panels_hidden = False

    def _on_view_type_change(self, view_type: str):
        """Handle view type change."""
        # Update posture panel display for new view type
        if hasattr(self, 'posture_panel'):
            self.posture_panel.update_view_type()

        # Re-process if we have results
        if self.batch_results:
            self.gait_control_panel.set_results(0, 0, 0, 0, 0)
            self.status_label.configure(
                text=f"View type changed to {view_type}. Re-process video to update results."
            )

    def _auto_detect_view_type(self):
        """Auto-detect view type from video content."""
        if not self.video_processor or not self.pose_detector:
            messagebox.showinfo("Info", "Load a video first")
            return

        self.status_label.configure(text="Analyzing view type...")
        self.root.update_idletasks()

        # Sample frames for analysis
        sample_frames = min(100, self.video_processor.total_frames)
        step = max(1, self.video_processor.total_frames // sample_frames)

        landmarks_sequence = []
        for i in range(0, self.video_processor.total_frames, step):
            frame = self.video_processor.get_frame(i)
            landmarks = self.pose_detector.detect(frame)
            if landmarks:
                landmarks_sequence.append(landmarks)

        if len(landmarks_sequence) < 10:
            messagebox.showwarning("Warning", "Not enough poses detected for view analysis")
            return

        # Analyze
        analyzer = ViewTypeAnalyzer()
        view_type, confidence = analyzer.analyze(landmarks_sequence)

        self.gait_control_panel.set_view_confidence(view_type, confidence)
        self.status_label.configure(
            text=f"Detected view type: {view_type} (confidence: {confidence:.0%})"
        )

    def _batch_process_video(self):
        """Process entire video for gait analysis."""
        if not self.video_processor or not self.pose_detector:
            messagebox.showinfo("Info", "Load a video first")
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.gait_control_panel.enable_processing(False)

        # Run processing in a thread to keep UI responsive
        def process():
            try:
                self.batch_results = batch_process_video(
                    self.video_processor,
                    self.pose_detector,
                    self.view_type.get(),
                    self.com_calculator,
                    progress_callback=self._update_processing_progress
                )

                # Normalize cycles
                self.normalized_cycles = []
                for cycle in self.batch_results.gait_cycles:
                    normalized = self.gait_normalizer.normalize_cycle(
                        cycle,
                        self.batch_results.frame_data,
                        self.batch_results.fps
                    )
                    if normalized:
                        self.normalized_cycles.append(normalized)

                # Update UI on main thread
                self.root.after(0, self._on_processing_complete)

            except Exception as e:
                self.root.after(0, lambda: self._on_processing_error(str(e)))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _update_processing_progress(self, current: int, total: int):
        """Update progress bar during batch processing."""
        self.root.after(0, lambda: self.gait_control_panel.set_progress(
            current, total, f"Processing frame {current}/{total}"
        ))

    def _on_processing_complete(self):
        """Handle processing completion."""
        self.is_processing = False
        self.gait_control_panel.enable_processing(True)

        if self.batch_results:
            num_cycles = len(self.batch_results.gait_cycles)
            num_events = len(self.batch_results.heel_strike_events)

            # Calculate mean step asymmetry
            asymmetries = [
                getattr(c, 'contralateral_timing', 50.0) - 50.0
                for c in self.batch_results.gait_cycles
            ]
            mean_asymmetry = sum(abs(a) for a in asymmetries) / len(asymmetries) if asymmetries else 0.0

            self.gait_control_panel.set_results(
                num_cycles, num_events,
                self.batch_results.processing_time_seconds,
                mean_asymmetry
            )

            self.gait_control_panel.set_progress(
                self.batch_results.total_frames,
                self.batch_results.total_frames,
                "Complete"
            )

            # Update timeline
            self.timeline_panel.set_data(
                self.batch_results.total_frames,
                self.batch_results.heel_strike_events,
                self.batch_results.gait_cycles
            )

            # Update cycles list
            self.cycles_panel.set_cycles(self.batch_results.gait_cycles)

            # Update comparison panel
            self.comparison_panel.set_data(self.normalized_cycles)

            # Update CoM panel
            self.com_panel.set_data(self.normalized_cycles)

            # Update Posture panel
            self.posture_panel.set_data(self.normalized_cycles)

            self.status_label.configure(
                text=f"Processing complete: {num_cycles} standard gait cycles detected"
            )

    def _on_processing_error(self, error_msg: str):
        """Handle processing error."""
        self.is_processing = False
        self.gait_control_panel.enable_processing(True)
        messagebox.showerror("Processing Error", f"Error during processing:\n{error_msg}")
        self.status_label.configure(text="Processing failed")

    def _add_heel_strike(self, frame_idx: int, side: str):
        """Add a manual heel strike event."""
        if not self.batch_results:
            messagebox.showinfo("Info", "Process video first to enable manual marking")
            return

        # Create new event
        new_event = HeelStrikeEvent(
            frame_idx=frame_idx,
            time_seconds=frame_idx / self.batch_results.fps,
            side=side,
            confidence=1.0,
            is_manual=True
        )

        # Add to events list
        self.batch_results.heel_strike_events.append(new_event)
        self.batch_results.heel_strike_events.sort(key=lambda e: e.frame_idx)

        # Rebuild cycles
        self._rebuild_cycles()

        self.status_label.configure(
            text=f"Added {side} heel strike at frame {frame_idx}"
        )

    def _remove_heel_strike(self, event: HeelStrikeEvent):
        """Remove a heel strike event."""
        if not self.batch_results:
            return

        try:
            self.batch_results.heel_strike_events.remove(event)
            self._rebuild_cycles()
            self.status_label.configure(
                text=f"Removed {event.side} heel strike at frame {event.frame_idx}"
            )
        except ValueError:
            pass

    def _clear_auto_detected(self):
        """Remove all auto-detected heel strike events (keep only manual ones)."""
        if not self.batch_results:
            return

        # Count events before removal
        original_count = len(self.batch_results.heel_strike_events)

        # Keep only manually marked events
        self.batch_results.heel_strike_events = [
            e for e in self.batch_results.heel_strike_events if e.is_manual
        ]

        removed_count = original_count - len(self.batch_results.heel_strike_events)

        # Rebuild cycles with remaining events
        self._rebuild_cycles()

        self.status_label.configure(
            text=f"Cleared {removed_count} auto-detected heel strikes. "
                 f"{len(self.batch_results.heel_strike_events)} manual events remain."
        )

    def _rebuild_cycles(self):
        """Rebuild gait cycles from current events."""
        if not self.batch_results:
            return

        # Create detector to rebuild cycles
        detector = GaitCycleDetector(
            self.batch_results.view_type,
            self.batch_results.fps
        )
        self.batch_results.gait_cycles = detector.build_cycles(
            self.batch_results.heel_strike_events
        )

        # Re-normalize
        self.normalized_cycles = []
        for cycle in self.batch_results.gait_cycles:
            normalized = self.gait_normalizer.normalize_cycle(
                cycle,
                self.batch_results.frame_data,
                self.batch_results.fps
            )
            if normalized:
                self.normalized_cycles.append(normalized)

        # Update UI
        self._update_gait_panels()

    def _update_gait_panels(self):
        """Update all gait analysis panels with current data."""
        if self.batch_results:
            num_cycles = len(self.batch_results.gait_cycles)
            num_events = len(self.batch_results.heel_strike_events)

            # Calculate mean step asymmetry
            asymmetries = [
                getattr(c, 'contralateral_timing', 50.0) - 50.0
                for c in self.batch_results.gait_cycles
            ]
            mean_asymmetry = sum(abs(a) for a in asymmetries) / len(asymmetries) if asymmetries else 0.0

            self.gait_control_panel.set_results(
                num_cycles, num_events,
                self.batch_results.processing_time_seconds,
                mean_asymmetry
            )

            self.timeline_panel.set_data(
                self.batch_results.total_frames,
                self.batch_results.heel_strike_events,
                self.batch_results.gait_cycles
            )

            self.cycles_panel.set_cycles(self.batch_results.gait_cycles)
            self.comparison_panel.set_data(self.normalized_cycles)
            self.posture_panel.set_data(self.normalized_cycles)
            self.com_panel.set_data(self.normalized_cycles)

    def _on_cycle_select(self, cycle: GaitCycle):
        """Handle cycle selection - navigate to start frame."""
        self._goto_frame(cycle.start_frame)

    def _export_data(self, format_type: str, path: str):
        """Export gait data."""
        if not self.batch_results or not self.normalized_cycles:
            messagebox.showinfo("Info", "No data to export. Process a video first.")
            return

        try:
            package = self.gait_exporter.create_export_package(
                self.batch_results,
                self.normalized_cycles
            )

            if format_type == 'numpy':
                self.gait_exporter.export_numpy(package, path)
            elif format_type == 'pickle':
                self.gait_exporter.export_pickle(package, path)
            elif format_type == 'all':
                # Remove extension for all-format export
                base_path = path.rsplit('.', 1)[0] if '.' in path else path
                self.gait_exporter.export_all(package, base_path)

            self.export_panel.set_status(f"Exported successfully to {path}")
            self.status_label.configure(text=f"Data exported to {path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")

    def _update_timeline_frame(self):
        """Update timeline with current frame position."""
        if hasattr(self, 'timeline_panel'):
            self.timeline_panel.set_current_frame(self.current_frame_idx)

    def close(self):
        """Clean up resources."""
        if self.video_processor:
            self.video_processor.close()
        if self.pose_detector:
            self.pose_detector.close()


def main():
    """Main entry point."""
    # Get video path from command line if provided
    video_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Create main window
    root = tk.Tk()
    root.geometry("1200x1200")  # Sized for portrait videos and gait analysis panels

    # Create application
    app = PoseViewerApp(root, video_path)

    # Handle window close
    def on_closing():
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()