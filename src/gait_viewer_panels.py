"""
Gait analysis UI panels for the pose viewer application.

Provides Tkinter-based UI components for gait cycle analysis,
including control panels, timeline visualization, and comparison plots.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, List, Dict, Optional, Any
import numpy as np

# Try to import matplotlib for plotting (optional)
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from gait_cycle import HeelStrikeEvent, GaitCycle
from gait_normalizer import NormalizedGaitData, ANGLE_NAMES


class GaitControlPanel(ttk.Frame):
    """
    Control panel for gait analysis settings and actions.

    Includes view type selection, batch processing controls,
    and manual heel strike marking tools.
    """

    def __init__(
        self,
        parent,
        on_process: Callable = None,
        on_view_change: Callable[[str], None] = None,
        on_auto_detect_view: Callable = None,
        view_type_var: tk.StringVar = None,
        **kwargs
    ):
        """
        Initialize the control panel.

        Args:
            parent: Parent widget
            on_process: Callback for batch processing
            on_view_change: Callback when view type changes
            on_auto_detect_view: Callback to auto-detect view type
            view_type_var: StringVar for view type ('side' or 'front')
        """
        super().__init__(parent, **kwargs)

        self.on_process = on_process
        self.on_view_change = on_view_change
        self.on_auto_detect_view = on_auto_detect_view
        self.view_type_var = view_type_var or tk.StringVar(value='side_right')

        self._setup_ui()

    def _setup_ui(self):
        """Create the control panel UI."""
        # View type selection frame
        view_frame = ttk.LabelFrame(self, text="Camera View")
        view_frame.pack(fill=tk.X, padx=5, pady=5)

        view_inner = ttk.Frame(view_frame)
        view_inner.pack(fill=tk.X, padx=5, pady=5)

        ttk.Radiobutton(
            view_inner,
            text="Left Side View",
            variable=self.view_type_var,
            value='side_left',
            command=self._on_view_type_change
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            view_inner,
            text="Right Side View",
            variable=self.view_type_var,
            value='side_right',
            command=self._on_view_type_change
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            view_inner,
            text="Front View",
            variable=self.view_type_var,
            value='front',
            command=self._on_view_type_change
        ).pack(side=tk.LEFT, padx=5)

        self.auto_detect_btn = ttk.Button(
            view_inner,
            text="Auto-Detect",
            command=self._on_auto_detect
        )
        self.auto_detect_btn.pack(side=tk.LEFT, padx=20)

        self.view_confidence_label = ttk.Label(view_inner, text="")
        self.view_confidence_label.pack(side=tk.LEFT, padx=5)

        # Processing controls frame
        process_frame = ttk.LabelFrame(self, text="Batch Processing")
        process_frame.pack(fill=tk.X, padx=5, pady=5)

        process_inner = ttk.Frame(process_frame)
        process_inner.pack(fill=tk.X, padx=5, pady=5)

        self.process_btn = ttk.Button(
            process_inner,
            text="Process Video",
            command=self._on_process
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            process_inner,
            variable=self.progress_var,
            maximum=100,
            length=200
        )
        self.progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.progress_label = ttk.Label(process_inner, text="Ready")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # Results summary frame
        self.results_frame = ttk.LabelFrame(self, text="Results")
        self.results_frame.pack(fill=tk.X, padx=5, pady=5)

        self.results_label = ttk.Label(
            self.results_frame,
            text="No results yet. Process a video to detect gait cycles."
        )
        self.results_label.pack(padx=5, pady=5)

    def _on_view_type_change(self):
        """Handle view type radio button change."""
        if self.on_view_change:
            self.on_view_change(self.view_type_var.get())

    def _on_auto_detect(self):
        """Handle auto-detect button click."""
        if self.on_auto_detect_view:
            self.on_auto_detect_view()

    def _on_process(self):
        """Handle process button click."""
        if self.on_process:
            self.on_process()

    def set_progress(self, current: int, total: int, message: str = None):
        """Update progress bar and label."""
        if total > 0:
            pct = 100 * current / total
            self.progress_var.set(pct)
            if message:
                self.progress_label.configure(text=message)
            else:
                self.progress_label.configure(text=f"{current}/{total} frames")
        self.update_idletasks()

    def set_results(self, num_cycles: int, num_events: int,
                   processing_time: float, mean_asymmetry: float = 0.0):
        """Update results display."""
        if num_events == 0:
            self.results_label.configure(
                text=f"Pose data extracted. Use Timeline tab to mark heel strikes manually.\n"
                     f"Processing time: {processing_time:.1f} seconds"
            )
        else:
            asymmetry_text = f"Mean step asymmetry: {mean_asymmetry:.1f}%"
            self.results_label.configure(
                text=f"{num_cycles} standard gait cycles (R HS → R HS) "
                     f"from {num_events} manually marked heel strikes.\n"
                     f"{asymmetry_text} | Processing time: {processing_time:.1f} seconds"
            )

    def set_view_confidence(self, view_type: str, confidence: float):
        """Update view type auto-detection confidence display."""
        self.view_type_var.set(view_type)
        self.view_confidence_label.configure(
            text=f"(confidence: {confidence:.0%})"
        )

    def enable_processing(self, enabled: bool = True):
        """Enable or disable processing button."""
        state = 'normal' if enabled else 'disabled'
        self.process_btn.configure(state=state)
        self.auto_detect_btn.configure(state=state)


class HeelStrikeTimelinePanel(ttk.Frame):
    """
    Timeline visualization panel for heel strikes and gait cycles.

    Displays a horizontal timeline with heel strike markers that can be
    clicked to navigate to frames, and supports manual marking.
    """

    def __init__(
        self,
        parent,
        on_frame_select: Callable[[int], None] = None,
        on_event_add: Callable[[int, str], None] = None,
        on_event_remove: Callable[[HeelStrikeEvent], None] = None,
        **kwargs
    ):
        """
        Initialize the timeline panel.

        Args:
            parent: Parent widget
            on_frame_select: Callback when a frame is selected
            on_event_add: Callback to add heel strike (frame_idx, side)
            on_event_remove: Callback to remove heel strike event
        """
        super().__init__(parent, **kwargs)

        self.on_frame_select = on_frame_select
        self.on_event_add = on_event_add
        self.on_event_remove = on_event_remove

        self.total_frames = 100
        self.events: List[HeelStrikeEvent] = []
        self.cycles: List[GaitCycle] = []
        self.current_frame = 0
        self.selected_event: Optional[HeelStrikeEvent] = None

        self._setup_ui()

    def _setup_ui(self):
        """Create the timeline UI."""
        # Manual marking controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="Manual Marking:").pack(side=tk.LEFT, padx=5)

        self.add_left_btn = ttk.Button(
            control_frame,
            text="Add Left HS",
            command=lambda: self._on_add_event('left')
        )
        self.add_left_btn.pack(side=tk.LEFT, padx=2)

        self.add_right_btn = ttk.Button(
            control_frame,
            text="Add Right HS",
            command=lambda: self._on_add_event('right')
        )
        self.add_right_btn.pack(side=tk.LEFT, padx=2)

        self.remove_btn = ttk.Button(
            control_frame,
            text="Remove Selected",
            command=self._on_remove_event,
            state='disabled'
        )
        self.remove_btn.pack(side=tk.LEFT, padx=10)

        # Timeline canvas
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, height=120, bg='#1e1e1e')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        # Legend
        legend_frame = ttk.Frame(self)
        legend_frame.pack(fill=tk.X, padx=5)

        ttk.Label(legend_frame, text="Legend:", foreground='gray').pack(side=tk.LEFT)
        ttk.Label(legend_frame, text=" Left HS", foreground='#ff4444').pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text=" Right HS", foreground='#4444ff').pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text=" Current Frame", foreground='#44ff44').pack(side=tk.LEFT, padx=5)

    def set_data(
        self,
        total_frames: int,
        events: List[HeelStrikeEvent],
        cycles: List[GaitCycle]
    ):
        """Set timeline data."""
        self.total_frames = max(1, total_frames)
        self.events = events
        self.cycles = cycles
        self._redraw()

    def set_current_frame(self, frame_idx: int):
        """Update current frame marker."""
        self.current_frame = frame_idx
        self._redraw()

    def _on_add_event(self, side: str):
        """Handle add event button click."""
        if self.on_event_add:
            self.on_event_add(self.current_frame, side)

    def _on_remove_event(self):
        """Handle remove event button click."""
        if self.selected_event and self.on_event_remove:
            self.on_event_remove(self.selected_event)
            self.selected_event = None
            self.remove_btn.configure(state='disabled')

    def _on_canvas_click(self, event):
        """Handle canvas click."""
        canvas_width = self.canvas.winfo_width()
        margin = 20

        # Check if click is within timeline area
        if margin <= event.x <= canvas_width - margin:
            # Calculate frame from x position
            x_ratio = (event.x - margin) / (canvas_width - 2 * margin)
            frame_idx = int(x_ratio * self.total_frames)
            frame_idx = max(0, min(frame_idx, self.total_frames - 1))

            # Check if clicking near an event marker
            clicked_event = None
            for evt in self.events:
                evt_x = margin + (evt.frame_idx / self.total_frames) * (canvas_width - 2 * margin)
                if abs(event.x - evt_x) < 10:
                    clicked_event = evt
                    break

            if clicked_event:
                self.selected_event = clicked_event
                self.remove_btn.configure(state='normal')
                self._redraw()
            else:
                self.selected_event = None
                self.remove_btn.configure(state='disabled')
                if self.on_frame_select:
                    self.on_frame_select(frame_idx)
                self._redraw()

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        self._redraw()

    def _redraw(self):
        """Redraw the timeline."""
        self.canvas.delete('all')

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        margin = 20

        if width < 100:
            return

        timeline_width = width - 2 * margin
        timeline_y = height // 2

        # Draw timeline background
        self.canvas.create_rectangle(
            margin, timeline_y - 2,
            width - margin, timeline_y + 2,
            fill='#444444', outline=''
        )

        # Draw cycle regions (all cycles are standard R HS → R HS)
        for i, cycle in enumerate(self.cycles):
            x1 = margin + (cycle.start_frame / self.total_frames) * timeline_width
            x2 = margin + (cycle.end_frame / self.total_frames) * timeline_width

            # Alternate colors for visibility, show asymmetry with intensity
            base_hue = i % 2  # Alternate between two colors
            asymmetry = abs(getattr(cycle, 'contralateral_timing', 50.0) - 50.0)
            # More asymmetric cycles get more red tint
            if asymmetry > 5:
                color = '#664433'  # Orange tint for asymmetric
            else:
                color = '#336666' if base_hue == 0 else '#335566'
            self.canvas.create_rectangle(
                x1, timeline_y - 15,
                x2, timeline_y + 15,
                fill=color, outline='',
                tags='cycle'
            )

        # Draw heel strike markers
        for evt in self.events:
            x = margin + (evt.frame_idx / self.total_frames) * timeline_width
            color = '#ff4444' if evt.side == 'left' else '#4444ff'

            # Draw marker
            if evt == self.selected_event:
                # Selected marker (larger, with highlight)
                self.canvas.create_polygon(
                    x, timeline_y - 25,
                    x - 8, timeline_y - 35,
                    x + 8, timeline_y - 35,
                    fill='yellow', outline='white'
                )
            else:
                # Normal marker
                self.canvas.create_polygon(
                    x, timeline_y - 20,
                    x - 6, timeline_y - 30,
                    x + 6, timeline_y - 30,
                    fill=color, outline='white'
                )

            # Manual marker indicator
            if evt.is_manual:
                self.canvas.create_text(
                    x, timeline_y - 40,
                    text='M', fill='white', font=('TkDefaultFont', 8)
                )

        # Draw current frame marker
        current_x = margin + (self.current_frame / self.total_frames) * timeline_width
        self.canvas.create_line(
            current_x, timeline_y - 40,
            current_x, timeline_y + 40,
            fill='#44ff44', width=2
        )

        # Draw frame numbers
        for i in range(0, self.total_frames + 1, max(1, self.total_frames // 10)):
            x = margin + (i / self.total_frames) * timeline_width
            self.canvas.create_text(
                x, timeline_y + 30,
                text=str(i), fill='gray', font=('TkDefaultFont', 8)
            )


class GaitCycleListPanel(ttk.Frame):
    """
    Panel listing all detected gait cycles with details.

    Provides a treeview for browsing cycles and selecting them for analysis.
    """

    def __init__(
        self,
        parent,
        on_cycle_select: Callable[[GaitCycle], None] = None,
        **kwargs
    ):
        """
        Initialize the cycle list panel.

        Args:
            parent: Parent widget
            on_cycle_select: Callback when a cycle is selected
        """
        super().__init__(parent, **kwargs)

        self.on_cycle_select = on_cycle_select
        self.cycles: List[GaitCycle] = []

        self._setup_ui()

    def _setup_ui(self):
        """Create the cycle list UI."""
        # Treeview with scrollbar
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ('id', 'start', 'end', 'duration', 'contra_timing', 'asymmetry', 'arm_amp', 'arm_vel')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=8)

        self.tree.heading('id', text='ID')
        self.tree.heading('start', text='Start Frame')
        self.tree.heading('end', text='End Frame')
        self.tree.heading('duration', text='Duration (s)')
        self.tree.heading('contra_timing', text='L HS Timing (%)')
        self.tree.heading('asymmetry', text='Asymmetry')
        self.tree.heading('arm_amp', text='Arm Swing (°)')
        self.tree.heading('arm_vel', text='Arm Vel (°/s)')

        self.tree.column('id', width=40)
        self.tree.column('start', width=80)
        self.tree.column('end', width=80)
        self.tree.column('duration', width=80)
        self.tree.column('contra_timing', width=100)
        self.tree.column('asymmetry', width=80)
        self.tree.column('arm_amp', width=90)
        self.tree.column('arm_vel', width=90)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind('<<TreeviewSelect>>', self._on_select)

        # Summary label
        self.summary_label = ttk.Label(self, text="No cycles detected")
        self.summary_label.pack(pady=5)

    def set_cycles(self, cycles: List[GaitCycle], normalized_cycles: List[NormalizedGaitData] = None):
        """Set the list of gait cycles.

        Args:
            cycles: List of GaitCycle objects
            normalized_cycles: Optional list of NormalizedGaitData for arm swing metrics
        """
        self.cycles = cycles

        # Build a lookup for normalized data by cycle_id
        norm_lookup = {}
        if normalized_cycles:
            for nc in normalized_cycles:
                norm_lookup[nc.cycle_id] = nc

        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Add new items
        asymmetries = []
        arm_amplitudes = []
        for cycle in cycles:
            contra_timing = getattr(cycle, 'contralateral_timing', 50.0)
            asymmetry = contra_timing - 50.0
            asymmetries.append(abs(asymmetry))

            # Get arm swing metrics from normalized data if available
            arm_amp = "--"
            arm_vel = "--"
            if cycle.cycle_id in norm_lookup:
                nc = norm_lookup[cycle.cycle_id]
                if nc.arm_swing_amplitude > 0:
                    arm_amp = f"{nc.arm_swing_amplitude:.1f}"
                    arm_amplitudes.append(nc.arm_swing_amplitude)
                    # Show peak velocity (larger absolute value)
                    peak_vel = max(abs(nc.arm_swing_peak_forward_velocity),
                                   abs(nc.arm_swing_peak_backward_velocity))
                    arm_vel = f"{peak_vel:.0f}"

            self.tree.insert('', 'end', values=(
                cycle.cycle_id,
                cycle.start_frame,
                cycle.end_frame,
                f"{cycle.duration_seconds:.2f}",
                f"{contra_timing:.1f}",
                f"{asymmetry:+.1f}%",
                arm_amp,
                arm_vel
            ))

        # Update summary
        mean_asymmetry = np.mean(asymmetries) if asymmetries else 0.0
        asymmetric_count = len([a for a in asymmetries if a > 5.0])
        arm_summary = ""
        if arm_amplitudes:
            mean_arm_amp = np.mean(arm_amplitudes)
            arm_summary = f" | Mean arm swing: {mean_arm_amp:.1f}°"

        self.summary_label.configure(
            text=f"Total: {len(cycles)} standard cycles | "
                 f"Mean asymmetry: {mean_asymmetry:.1f}% | "
                 f"Asymmetric (>5%): {asymmetric_count}{arm_summary}"
        )

    def _on_select(self, event):
        """Handle treeview selection."""
        selection = self.tree.selection()
        if selection and self.on_cycle_select:
            item = self.tree.item(selection[0])
            cycle_id = item['values'][0]

            # Find the cycle
            for cycle in self.cycles:
                if cycle.cycle_id == cycle_id:
                    self.on_cycle_select(cycle)
                    break


class CycleComparisonPanel(ttk.Frame):
    """
    Panel for comparing gait cycles with matplotlib plots.

    Displays normalized gait cycle data with options to compare
    left vs right and across multiple cycles.
    """

    def __init__(self, parent, **kwargs):
        """Initialize the comparison panel."""
        super().__init__(parent, **kwargs)

        self.normalized_cycles: List[NormalizedGaitData] = []

        self._setup_ui()

    def _setup_ui(self):
        """Create the comparison panel UI."""
        if not HAS_MATPLOTLIB:
            ttk.Label(
                self,
                text="Matplotlib not available. Install matplotlib for plotting."
            ).pack(padx=10, pady=10)
            return

        # Controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="Angle:").pack(side=tk.LEFT, padx=5)

        self.angle_var = tk.StringVar(value='knee')
        # Sagittal view angles + frontal view limb angles
        angle_options = [
            'shoulder', 'elbow', 'hip', 'knee', 'ankle', 'arm_swing',  # Sagittal
            'arm', 'leg'  # Frontal (shoulder-wrist and hip-ankle vs vertical)
        ]
        self.angle_combo = ttk.Combobox(
            control_frame,
            textvariable=self.angle_var,
            values=angle_options,
            state='readonly',
            width=12
        )
        self.angle_combo.pack(side=tk.LEFT, padx=5)
        self.angle_combo.bind('<<ComboboxSelected>>', lambda e: self._update_plot())

        ttk.Label(control_frame, text="Compare:").pack(side=tk.LEFT, padx=15)

        self.compare_var = tk.StringVar(value='left_vs_right')
        ttk.Radiobutton(
            control_frame,
            text="Left vs Right (bilateral)",
            variable=self.compare_var,
            value='left_vs_right',
            command=self._update_plot
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            control_frame,
            text="All Cycles (Left limb)",
            variable=self.compare_var,
            value='all_left',
            command=self._update_plot
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            control_frame,
            text="All Cycles (Right limb)",
            variable=self.compare_var,
            value='all_right',
            command=self._update_plot
        ).pack(side=tk.LEFT, padx=5)

        # Matplotlib figure - increased height for better label visibility
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.figure.patch.set_facecolor('#2b2b2b')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._setup_empty_plot()

    def _setup_empty_plot(self):
        """Set up empty plot with labels."""
        if not HAS_MATPLOTLIB:
            return

        self.ax.clear()
        self.ax.set_xlabel('Gait Cycle (%)', color='white', fontsize=10)
        self.ax.set_ylabel('Angle (degrees)', color='white', fontsize=10)
        self.ax.set_title('No data - Process video first', color='white', fontsize=11, pad=10)
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        try:
            self.figure.tight_layout()
        except ValueError:
            self.figure.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
        self.canvas.draw()

    def set_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Set normalized cycle data."""
        self.normalized_cycles = normalized_cycles
        self._update_plot()

    def _update_plot(self):
        """Update the comparison plot."""
        if not HAS_MATPLOTLIB or not self.normalized_cycles:
            return

        self.ax.clear()

        angle_base = self.angle_var.get()
        compare_mode = self.compare_var.get()

        x = np.arange(100)  # 0-99%

        if compare_mode == 'left_vs_right':
            self._plot_left_vs_right(angle_base, x)
        elif compare_mode == 'all_left':
            self._plot_all_cycles('left', angle_base, x)
        else:
            self._plot_all_cycles('right', angle_base, x)

        # Style
        self.ax.set_xlabel('Gait Cycle (%)', color='white', fontsize=10)
        self.ax.set_ylabel(f'{angle_base.capitalize()} Angle (degrees)', color='white', fontsize=10)
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.legend(facecolor='#2b2b2b', labelcolor='white')
        self.ax.grid(True, alpha=0.3)

        try:
            self.figure.tight_layout()
        except ValueError:
            self.figure.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
        self.canvas.draw()

    def _plot_left_vs_right(self, angle_base: str, x: np.ndarray):
        """Plot left vs right comparison with mean and std bands."""
        # Handle arm_swing specially - it's not bilateral
        if angle_base == 'arm_swing':
            self._plot_arm_swing(x)
            return

        left_angle = f'left_{angle_base}'
        right_angle = f'right_{angle_base}'

        # Collect data
        left_data = []
        right_data = []

        for cycle in self.normalized_cycles:
            try:
                left_series = cycle.get_angle_series(left_angle)
                right_series = cycle.get_angle_series(right_angle)

                if not np.all(np.isnan(left_series)):
                    left_data.append(left_series)
                if not np.all(np.isnan(right_series)):
                    right_data.append(right_series)
            except (ValueError, KeyError):
                continue

        # Plot left
        if left_data:
            left_stack = np.stack(left_data, axis=0)
            left_mean = np.nanmean(left_stack, axis=0)
            left_std = np.nanstd(left_stack, axis=0)

            self.ax.plot(x, left_mean, 'r-', linewidth=2, label=f'Left (n={len(left_data)})')
            self.ax.fill_between(x, left_mean - left_std, left_mean + left_std,
                                alpha=0.3, color='red')

        # Plot right
        if right_data:
            right_stack = np.stack(right_data, axis=0)
            right_mean = np.nanmean(right_stack, axis=0)
            right_std = np.nanstd(right_stack, axis=0)

            self.ax.plot(x, right_mean, 'b-', linewidth=2, label=f'Right (n={len(right_data)})')
            self.ax.fill_between(x, right_mean - right_std, right_mean + right_std,
                                alpha=0.3, color='blue')

        # Set appropriate title based on angle type
        if angle_base == 'arm':
            title = 'Arm Angle (vs Vertical): Left vs Right'
        elif angle_base == 'leg':
            title = 'Leg Angle (vs Vertical): Left vs Right'
        else:
            title = f'{angle_base.capitalize()} Angle: Left vs Right'
        self.ax.set_title(title, color='white', fontsize=11, pad=10)

    def _plot_arm_swing(self, x: np.ndarray):
        """Plot arm swing angle across all cycles with summary metrics."""
        cycles_data = []
        amplitudes = []
        peak_fwd_vels = []
        peak_bwd_vels = []

        for cycle in self.normalized_cycles:
            try:
                series = cycle.get_angle_series('arm_swing_angle')
                if not np.all(np.isnan(series)):
                    cycles_data.append((cycle.cycle_id, series))
                    amplitudes.append(cycle.arm_swing_amplitude)
                    peak_fwd_vels.append(cycle.arm_swing_peak_forward_velocity)
                    peak_bwd_vels.append(cycle.arm_swing_peak_backward_velocity)
            except (ValueError, KeyError):
                continue

        if not cycles_data:
            self.ax.set_title('Arm Swing: No data (use lateral view)', color='white', fontsize=11, pad=10)
            return

        # Plot individual cycles with light lines
        for cycle_id, series in cycles_data:
            self.ax.plot(x, series, alpha=0.4, linewidth=1, color='cyan')

        # Plot mean with bold line
        data_stack = np.stack([d[1] for d in cycles_data], axis=0)
        mean = np.nanmean(data_stack, axis=0)
        std = np.nanstd(data_stack, axis=0)

        self.ax.plot(x, mean, 'g-', linewidth=2.5, label=f'Mean (n={len(cycles_data)})')
        self.ax.fill_between(x, mean - std, mean + std, alpha=0.3, color='green')

        # Add zero reference line
        self.ax.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)

        # Add summary text box
        mean_amp = np.mean(amplitudes) if amplitudes else 0
        mean_fwd = np.mean(peak_fwd_vels) if peak_fwd_vels else 0
        mean_bwd = np.mean(peak_bwd_vels) if peak_bwd_vels else 0

        summary_text = (
            f'Amplitude: {mean_amp:.1f}°\n'
            f'Peak Fwd Vel: {mean_fwd:.1f}°/s\n'
            f'Peak Bwd Vel: {mean_bwd:.1f}°/s'
        )
        self.ax.text(0.98, 0.98, summary_text, transform=self.ax.transAxes,
                    fontsize=9, color='white', verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

        self.ax.set_title('Arm Swing Angle (relative to trunk)', color='white', fontsize=11, pad=10)
        self.ax.set_ylabel('Arm Swing Angle (degrees)', color='white', fontsize=10)

    def _plot_all_cycles(self, limb_side: str, angle_base: str, x: np.ndarray):
        """Plot the specified limb angle across all standard gait cycles."""
        # Handle arm_swing specially - redirect to _plot_arm_swing
        if angle_base == 'arm_swing':
            self._plot_arm_swing(x)
            return

        angle_name = f'{limb_side}_{angle_base}'
        color = 'red' if limb_side == 'left' else 'blue'

        cycles_data = []
        for cycle in self.normalized_cycles:
            try:
                series = cycle.get_angle_series(angle_name)
                if not np.all(np.isnan(series)):
                    cycles_data.append((cycle.cycle_id, series))
            except (ValueError, KeyError):
                continue

        # Plot individual cycles
        for cycle_id, series in cycles_data:
            self.ax.plot(x, series, alpha=0.5, linewidth=1)

        # Plot mean
        if cycles_data:
            data_stack = np.stack([d[1] for d in cycles_data], axis=0)
            mean = np.nanmean(data_stack, axis=0)
            self.ax.plot(x, mean, color=color, linewidth=3, label=f'Mean (n={len(cycles_data)})')

        self.ax.set_title(f'{limb_side.capitalize()} {angle_base.capitalize()} Angle: All Cycles', color='white', fontsize=11, pad=10)


class CenterOfMassPanel(ttk.Frame):
    """
    Panel for visualizing center of mass trajectory.

    Displays CoM path and time series plots.
    """

    def __init__(self, parent, **kwargs):
        """Initialize the CoM panel."""
        super().__init__(parent, **kwargs)

        self.normalized_cycles: List[NormalizedGaitData] = []

        self._setup_ui()

    def _setup_ui(self):
        """Create the CoM panel UI."""
        if not HAS_MATPLOTLIB:
            ttk.Label(
                self,
                text="Matplotlib not available. Install matplotlib for plotting."
            ).pack(padx=10, pady=10)
            return

        # Controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="Display:").pack(side=tk.LEFT, padx=5)

        self.display_var = tk.StringVar(value='x_time')
        ttk.Radiobutton(
            control_frame,
            text="X Position (horizontal) vs Gait Cycle",
            variable=self.display_var,
            value='x_time',
            command=self._update_plot
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            control_frame,
            text="Y Position (vertical) vs Gait Cycle",
            variable=self.display_var,
            value='y_time',
            command=self._update_plot
        ).pack(side=tk.LEFT, padx=5)

        # Matplotlib figure - increased height for better label visibility
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.figure.patch.set_facecolor('#2b2b2b')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._setup_empty_plot()

    def _setup_empty_plot(self):
        """Set up empty plot."""
        if not HAS_MATPLOTLIB:
            return

        self.ax.clear()
        self.ax.set_xlabel('Gait Cycle (%)', color='white', fontsize=10)
        self.ax.set_ylabel('Position (normalized)', color='white', fontsize=10)
        self.ax.set_title('No data - Process video first', color='white', fontsize=11, pad=10)
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        try:
            self.figure.tight_layout()
        except ValueError:
            # Fallback if tight_layout fails
            self.figure.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
        self.canvas.draw()

    def set_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Set normalized cycle data."""
        self.normalized_cycles = normalized_cycles
        self._update_plot()

    def _update_plot(self):
        """Update the CoM plot."""
        if not HAS_MATPLOTLIB or not self.normalized_cycles:
            return

        self.ax.clear()

        display_mode = self.display_var.get()

        if display_mode == 'x_time':
            self._plot_position_time('x')
        else:
            self._plot_position_time('y')

        # Style
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        try:
            self.figure.tight_layout()
        except ValueError:
            self.figure.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
        self.canvas.draw()

    def _plot_position_time(self, axis: str):
        """Plot CoM position over normalized time for all standard cycles."""
        idx = 0 if axis == 'x' else 1
        x = np.arange(100)

        # Collect data from all cycles
        all_data = []
        symmetric_data = []
        asymmetric_data = []

        for cycle in self.normalized_cycles:
            com = cycle.center_of_mass[:, idx]
            if not np.all(np.isnan(com)):
                all_data.append(com)
                # Classify by symmetry
                asym = abs(getattr(cycle, 'contralateral_timing', 50.0) - 50.0)
                if asym <= 5.0:
                    symmetric_data.append(com)
                else:
                    asymmetric_data.append(com)

        # Plot all cycles mean with confidence band
        if all_data:
            all_stack = np.stack(all_data, axis=0)
            all_mean = np.nanmean(all_stack, axis=0)
            all_std = np.nanstd(all_stack, axis=0)

            self.ax.plot(x, all_mean, 'c-', linewidth=2, label=f'All cycles (n={len(all_data)})')
            self.ax.fill_between(x, all_mean - all_std, all_mean + all_std,
                                alpha=0.3, color='cyan')

        # Mark contralateral HS zone (around 50%)
        self.ax.axvline(x=50, color='yellow', linestyle='--', alpha=0.5, label='Expected L HS (50%)')

        self.ax.set_xlabel('Gait Cycle (%) - R HS to R HS', color='white', fontsize=10)
        self.ax.set_ylabel(f'CoM {axis.upper()} Position (normalized)', color='white', fontsize=10)
        self.ax.set_title(f'Center of Mass {axis.upper()} Position Over Gait Cycle', color='white', fontsize=11, pad=10)
        self.ax.legend(facecolor='#2b2b2b', labelcolor='white')
        self.ax.grid(True, alpha=0.3)


class PosturalAnglesPanel(ttk.Frame):
    """
    Panel for visualizing postural/spinal angles over the gait cycle.

    Displays view-dependent postural angles:
    - Side view (Sagittal): cervical flexion, thoracic inclination, trunk inclination
    - Front view (Frontal): shoulder tilt, hip/pelvic tilt, trunk lateral lean
    """

    # Postural angle definitions by view type
    SAGITTAL_ANGLES = [
        ('cervical_flexion', 'Cervical Flexion', '#ff6b6b'),
        ('thoracic_inclination', 'Thoracic Inclination', '#4ecdc4'),
        ('trunk_inclination', 'Trunk Inclination', '#45b7d1'),
    ]

    FRONTAL_ANGLES = [
        ('shoulder_tilt', 'Shoulder Tilt', '#ff6b6b'),
        ('hip_tilt', 'Pelvic Tilt', '#4ecdc4'),
        ('trunk_lateral_lean', 'Trunk Lateral Lean', '#45b7d1'),
    ]

    def __init__(self, parent, view_type_var: tk.StringVar = None, **kwargs):
        """
        Initialize the postural angles panel.

        Args:
            parent: Parent widget
            view_type_var: StringVar for current view type ('side' or 'front')
        """
        super().__init__(parent, **kwargs)

        self.normalized_cycles: List[NormalizedGaitData] = []
        self.view_type_var = view_type_var or tk.StringVar(value='side')

        self._setup_ui()

    def _setup_ui(self):
        """Create the postural angles panel UI."""
        if not HAS_MATPLOTLIB:
            ttk.Label(
                self,
                text="Matplotlib not available. Install matplotlib for plotting."
            ).pack(padx=10, pady=10)
            return

        # Info label showing current view type
        info_frame = ttk.Frame(self)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.view_info_label = ttk.Label(
            info_frame,
            text="Showing: Sagittal plane angles (Side View)"
        )
        self.view_info_label.pack(side=tk.LEFT, padx=5)

        # Display options
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="Display:").pack(side=tk.LEFT, padx=5)

        self.display_var = tk.StringVar(value='all')
        ttk.Radiobutton(
            control_frame,
            text="All Angles",
            variable=self.display_var,
            value='all',
            command=self._update_plot
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            control_frame,
            text="Individual Cycles",
            variable=self.display_var,
            value='individual',
            command=self._update_plot
        ).pack(side=tk.LEFT, padx=5)

        # Matplotlib figure - increased height for better label visibility
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.figure.patch.set_facecolor('#2b2b2b')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._setup_empty_plot()

    def _setup_empty_plot(self):
        """Set up empty plot with labels."""
        if not HAS_MATPLOTLIB:
            return

        self.ax.clear()
        self.ax.set_xlabel('Gait Cycle (%)', color='white')
        self.ax.set_ylabel('Angle (degrees)', color='white')
        self.ax.set_title('No data - Process video first', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        try:
            self.figure.tight_layout()
        except ValueError:
            self.figure.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
        self.canvas.draw()

    def set_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Set normalized cycle data."""
        self.normalized_cycles = normalized_cycles
        self._update_plot()

    def update_view_type(self):
        """Update display when view type changes."""
        self._update_plot()

    def _get_current_angles(self) -> List[tuple]:
        """Get the angle definitions for the current view type."""
        view_type = self.view_type_var.get()
        if view_type == 'front':
            return self.FRONTAL_ANGLES
        else:
            return self.SAGITTAL_ANGLES

    def _update_plot(self):
        """Update the postural angles plot."""
        if not HAS_MATPLOTLIB:
            return

        self.ax.clear()

        view_type = self.view_type_var.get()
        angles = self._get_current_angles()

        # Update info label
        if view_type == 'front':
            self.view_info_label.configure(
                text="Showing: Frontal plane angles (Front View) - Shoulder Tilt, Pelvic Tilt, Lateral Lean"
            )
        else:
            self.view_info_label.configure(
                text="Showing: Sagittal plane angles (Side View) - Cervical, Thoracic, Trunk Inclination"
            )

        if not self.normalized_cycles:
            self._setup_empty_plot()
            return

        x = np.arange(100)  # 0-99%
        display_mode = self.display_var.get()

        for angle_name, display_name, color in angles:
            self._plot_angle(angle_name, display_name, color, x, display_mode)

        # Style
        self.ax.set_xlabel('Gait Cycle (%)', color='white', fontsize=10)
        self.ax.set_ylabel('Angle (degrees)', color='white', fontsize=10)

        if view_type == 'front':
            title = 'Frontal Plane Postural Angles Over Gait Cycle'
        else:
            title = 'Sagittal Plane Postural Angles Over Gait Cycle'

        self.ax.set_title(title, color='white', fontsize=11, pad=10)
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.legend(facecolor='#2b2b2b', labelcolor='white', loc='best')
        self.ax.grid(True, alpha=0.3)

        # Add zero reference line for tilt angles
        self.ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)

        # Mark contralateral heel strike (around 50%)
        self.ax.axvline(x=50, color='yellow', linestyle='--', alpha=0.4,
                       label='Expected L HS')

        try:
            self.figure.tight_layout()
        except ValueError:
            self.figure.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
        self.canvas.draw()

    def _plot_angle(self, angle_name: str, display_name: str, color: str,
                    x: np.ndarray, display_mode: str):
        """Plot a single postural angle with mean and std bands."""
        # Collect data from all cycles
        angle_data = []

        for cycle in self.normalized_cycles:
            try:
                series = cycle.get_angle_series(angle_name)
                if not np.all(np.isnan(series)):
                    angle_data.append(series)
            except (ValueError, KeyError):
                continue

        if not angle_data:
            return

        data_stack = np.stack(angle_data, axis=0)

        if display_mode == 'individual':
            # Plot individual cycles with transparency
            for i, series in enumerate(angle_data):
                self.ax.plot(x, series, color=color, alpha=0.3, linewidth=1)

        # Always plot mean
        mean = np.nanmean(data_stack, axis=0)
        std = np.nanstd(data_stack, axis=0)

        self.ax.plot(x, mean, color=color, linewidth=2,
                    label=f'{display_name} (n={len(angle_data)})')

        # Add std band
        self.ax.fill_between(x, mean - std, mean + std,
                            alpha=0.2, color=color)


class ExportPanel(ttk.Frame):
    """
    Panel for exporting gait analysis data.
    """

    def __init__(
        self,
        parent,
        on_export: Callable[[str, str], None] = None,
        **kwargs
    ):
        """
        Initialize the export panel.

        Args:
            parent: Parent widget
            on_export: Callback(format, path) for export
        """
        super().__init__(parent, **kwargs)

        self.on_export = on_export
        self._setup_ui()

    def _setup_ui(self):
        """Create the export panel UI."""
        ttk.Label(self, text="Export gait cycle data:").pack(anchor=tk.W, padx=5, pady=5)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            btn_frame,
            text="Export NumPy (.npz)",
            command=lambda: self._export('numpy')
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="Export Pickle (.pkl)",
            command=lambda: self._export('pickle')
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="Export All Formats",
            command=lambda: self._export('all')
        ).pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = ttk.Label(self, text="")
        self.status_label.pack(anchor=tk.W, padx=5, pady=5)

    def _export(self, format_type: str):
        """Handle export button click."""
        if format_type == 'numpy':
            filetypes = [("NumPy archive", "*.npz")]
            default_ext = '.npz'
        elif format_type == 'pickle':
            filetypes = [("Pickle file", "*.pkl")]
            default_ext = '.pkl'
        else:
            filetypes = [("All files", "*.*")]
            default_ext = ''

        path = filedialog.asksaveasfilename(
            title="Export Gait Data",
            filetypes=filetypes,
            defaultextension=default_ext
        )

        if path and self.on_export:
            self.on_export(format_type, path)

    def set_status(self, message: str):
        """Set status message."""
        self.status_label.configure(text=message)


# =============================================================================
# Dual-View Wrapper Panels (Sagittal/Frontal sub-tabs)
# =============================================================================

class DualViewCycleListPanel(ttk.Frame):
    """
    Wrapper panel with Sagittal/Frontal sub-tabs for gait cycle lists.
    """

    def __init__(
        self,
        parent,
        on_cycle_select: Callable[[GaitCycle], None] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)

        self.on_cycle_select = on_cycle_select

        # Create sub-notebook for Sagittal/Frontal views
        self.sub_notebook = ttk.Notebook(self)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True)

        # Sagittal cycles panel
        self.sagittal_panel = GaitCycleListPanel(
            self.sub_notebook,
            on_cycle_select=on_cycle_select
        )
        self.sub_notebook.add(self.sagittal_panel, text="Sagittal")

        # Frontal cycles panel (simpler - no arm swing columns needed)
        self.frontal_panel = GaitCycleListPanel(
            self.sub_notebook,
            on_cycle_select=on_cycle_select
        )
        self.sub_notebook.add(self.frontal_panel, text="Frontal")

    def set_sagittal_data(self, cycles: List[GaitCycle],
                          normalized_cycles: List[NormalizedGaitData] = None):
        """Set sagittal view cycle data."""
        self.sagittal_panel.set_cycles(cycles, normalized_cycles)

    def set_frontal_data(self, cycles: List[GaitCycle],
                         normalized_cycles: List[NormalizedGaitData] = None):
        """Set frontal view cycle data."""
        self.frontal_panel.set_cycles(cycles, normalized_cycles)

    def set_cycles(self, cycles: List[GaitCycle],
                   normalized_cycles: List[NormalizedGaitData] = None):
        """Backward compatibility - sets sagittal data."""
        self.set_sagittal_data(cycles, normalized_cycles)


class DualViewComparisonPanel(ttk.Frame):
    """
    Wrapper panel with Sagittal/Frontal sub-tabs for cycle comparison plots.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Create sub-notebook for Sagittal/Frontal views
        self.sub_notebook = ttk.Notebook(self)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True)

        # Sagittal comparison panel
        self.sagittal_panel = CycleComparisonPanel(self.sub_notebook)
        self.sub_notebook.add(self.sagittal_panel, text="Sagittal")

        # Frontal comparison panel
        self.frontal_panel = CycleComparisonPanel(self.sub_notebook)
        self.sub_notebook.add(self.frontal_panel, text="Frontal")

    def set_sagittal_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Set sagittal view comparison data."""
        self.sagittal_panel.set_data(normalized_cycles)

    def set_frontal_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Set frontal view comparison data."""
        self.frontal_panel.set_data(normalized_cycles)

    def set_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Backward compatibility - sets sagittal data."""
        self.set_sagittal_data(normalized_cycles)


class DualViewPosturePanel(ttk.Frame):
    """
    Wrapper panel with Sagittal/Frontal sub-tabs for postural angles.
    """

    def __init__(self, parent, view_type_var: tk.StringVar = None, **kwargs):
        super().__init__(parent, **kwargs)

        # Store view type var
        self.view_type_var = view_type_var

        # Create sub-notebook for Sagittal/Frontal views
        self.sub_notebook = ttk.Notebook(self)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True)

        # Sagittal posture panel (uses 'side' view type for sagittal angles)
        sagittal_view_var = tk.StringVar(value='side_right')
        self.sagittal_panel = PosturalAnglesPanel(
            self.sub_notebook,
            view_type_var=sagittal_view_var
        )
        self.sub_notebook.add(self.sagittal_panel, text="Sagittal")

        # Frontal posture panel (uses 'front' view type for frontal angles)
        frontal_view_var = tk.StringVar(value='front')
        self.frontal_panel = PosturalAnglesPanel(
            self.sub_notebook,
            view_type_var=frontal_view_var
        )
        self.sub_notebook.add(self.frontal_panel, text="Frontal")

    def set_sagittal_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Set sagittal view posture data."""
        self.sagittal_panel.set_data(normalized_cycles)

    def set_frontal_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Set frontal view posture data."""
        self.frontal_panel.set_data(normalized_cycles)

    def set_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Backward compatibility - sets sagittal data."""
        self.set_sagittal_data(normalized_cycles)

    def update_view_type(self):
        """Update both panels when view type changes."""
        self.sagittal_panel.update_view_type()
        self.frontal_panel.update_view_type()


class DualViewCenterOfMassPanel(ttk.Frame):
    """
    Wrapper panel with Sagittal/Frontal sub-tabs for center of mass visualization.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Create sub-notebook for Sagittal/Frontal views
        self.sub_notebook = ttk.Notebook(self)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True)

        # Sagittal CoM panel
        self.sagittal_panel = CenterOfMassPanel(self.sub_notebook)
        self.sub_notebook.add(self.sagittal_panel, text="Sagittal")

        # Frontal CoM panel
        self.frontal_panel = CenterOfMassPanel(self.sub_notebook)
        self.sub_notebook.add(self.frontal_panel, text="Frontal")

    def set_sagittal_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Set sagittal view CoM data."""
        self.sagittal_panel.set_data(normalized_cycles)

    def set_frontal_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Set frontal view CoM data."""
        self.frontal_panel.set_data(normalized_cycles)

    def set_data(self, normalized_cycles: List[NormalizedGaitData]):
        """Backward compatibility - sets sagittal data."""
        self.set_sagittal_data(normalized_cycles)
