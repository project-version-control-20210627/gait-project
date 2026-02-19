"""
Gait cycle management module.

Provides gait cycle building from manually marked heel strikes, and view type
analysis for both sagittal (side) and frontal (front) camera views.

Heel strikes are marked manually by the user through the Timeline panel in
the pose viewer application.
"""

from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HeelStrikeEvent:
    """Represents a heel strike event detected or marked in the video."""
    frame_idx: int
    time_seconds: float
    side: str  # 'left' or 'right'
    confidence: float  # 0.0 to 1.0
    is_manual: bool = False  # True if manually marked/corrected

    def __lt__(self, other):
        return self.frame_idx < other.frame_idx


@dataclass
class GaitCycle:
    """
    Represents one complete standard gait cycle.

    A standard gait cycle is defined from right heel strike to the next right heel strike.
    This allows direct comparison of bilateral joint angles at the same phase of the cycle.
    The contralateral (left) heel strike typically occurs at ~50% of the cycle.
    """
    start_event: HeelStrikeEvent          # Right heel strike (0% of cycle)
    end_event: HeelStrikeEvent            # Next right heel strike (100% of cycle)
    contralateral_event: HeelStrikeEvent  # Left heel strike within cycle (~50%)
    cycle_id: int
    contralateral_timing: float = 0.0     # Normalized timing (0-100%) of left HS
    frame_indices: List[int] = field(default_factory=list)
    duration_seconds: float = 0.0

    def __post_init__(self):
        if not self.frame_indices:
            self.frame_indices = list(range(self.start_event.frame_idx,
                                            self.end_event.frame_idx + 1))
        if self.duration_seconds == 0.0:
            self.duration_seconds = self.end_event.time_seconds - self.start_event.time_seconds
        # Calculate contralateral timing as percentage of cycle
        if self.contralateral_timing == 0.0 and self.duration_seconds > 0:
            contra_time = self.contralateral_event.time_seconds - self.start_event.time_seconds
            self.contralateral_timing = (contra_time / self.duration_seconds) * 100.0

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    @property
    def start_frame(self) -> int:
        return self.start_event.frame_idx

    @property
    def end_frame(self) -> int:
        return self.end_event.frame_idx

    @property
    def contralateral_frame(self) -> int:
        """Frame index of the contralateral (left) heel strike."""
        return self.contralateral_event.frame_idx

    @property
    def step_time_asymmetry(self) -> float:
        """
        Step time asymmetry: deviation of contralateral timing from 50%.

        Positive values indicate left heel strike occurs later than expected.
        Negative values indicate left heel strike occurs earlier than expected.
        Healthy gait typically has values close to 0.
        """
        return self.contralateral_timing - 50.0


@dataclass
class FramePoseData:
    """Pose data for a single frame."""
    frame_idx: int
    timestamp: float
    landmarks: Dict[str, Tuple[float, float, float]]  # name -> (x, y, visibility)
    joint_angles: Dict[str, Optional[float]] = field(default_factory=dict)
    center_of_mass: Optional[Tuple[float, float]] = None


@dataclass
class SubjectMetadata:
    """Metadata about the subject and recording session."""
    subject_id: str = ""
    date: str = ""  # ISO format: YYYY-MM-DD
    walking_condition: str = ""  # e.g., "barefoot", "with AFO", "post-surgery week 4"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for export."""
        return {
            "subject_id": self.subject_id,
            "date": self.date,
            "walking_condition": self.walking_condition
        }


@dataclass
class DualViewSession:
    """
    Container for a complete gait analysis session with both views.

    Holds the results from both sagittal (side) and frontal (front) video
    analysis, along with subject metadata.
    """
    metadata: SubjectMetadata
    sagittal_camera_side: str  # 'left' or 'right' - which side camera was on
    sagittal_results: 'BatchProcessingResults' = None
    frontal_results: 'BatchProcessingResults' = None
    sagittal_normalized_cycles: List['NormalizedGaitData'] = field(default_factory=list)
    frontal_normalized_cycles: List['NormalizedGaitData'] = field(default_factory=list)


class ViewTypeAnalyzer:
    """
    Analyzes motion patterns to determine if video is side view or front view.

    Side view (sagittal plane): Large horizontal motion of limbs during gait
    Front view (frontal plane): Lateral sway visible, shoulders appear at full width
    """

    def analyze(
        self,
        landmarks_sequence: List[Dict[str, Tuple[float, float, float]]],
        sample_rate: int = 1
    ) -> Tuple[str, float]:
        """
        Analyze landmark sequence to determine view type.

        Args:
            landmarks_sequence: List of landmark dictionaries for consecutive frames
            sample_rate: Sample every Nth frame (for efficiency on long videos)

        Returns:
            Tuple of (view_type, confidence) where view_type is 'side' or 'front'
        """
        # Sample frames for efficiency
        sampled = landmarks_sequence[::sample_rate]
        if len(sampled) < 10:
            return ('side', 0.5)  # Default to side view if not enough data

        # Calculate motion characteristics
        ankle_x_var = self._calculate_variance(sampled, 'left_ankle', 0)  # x-axis
        ankle_y_var = self._calculate_variance(sampled, 'left_ankle', 1)  # y-axis

        # Calculate apparent shoulder width (narrow in side view, wide in front)
        shoulder_widths = []
        for lm in sampled:
            if lm and lm.get('left_shoulder') and lm.get('right_shoulder'):
                left_vis = lm['left_shoulder'][2]
                right_vis = lm['right_shoulder'][2]
                if left_vis > 0.3 and right_vis > 0.3:
                    width = abs(lm['left_shoulder'][0] - lm['right_shoulder'][0])
                    shoulder_widths.append(width)

        avg_shoulder_width = np.mean(shoulder_widths) if shoulder_widths else 0.1

        # Decision logic
        # Side view: ankle moves more in x (walking direction), narrow shoulders
        # Front view: ankle moves more in y (vertical), wide shoulders

        x_to_y_ratio = ankle_x_var / (ankle_y_var + 1e-6)

        if x_to_y_ratio > 2.0 and avg_shoulder_width < 0.15:
            confidence = min(0.95, 0.5 + x_to_y_ratio / 10)
            return ('side', confidence)
        elif x_to_y_ratio < 0.5 and avg_shoulder_width > 0.20:
            confidence = min(0.95, 0.5 + (1 / (x_to_y_ratio + 0.1)) / 10)
            return ('front', confidence)
        else:
            # Ambiguous - lean towards side view but with low confidence
            if x_to_y_ratio > 1:
                return ('side', 0.5)
            else:
                return ('front', 0.5)

    def _calculate_variance(
        self,
        landmarks_sequence: List[Dict],
        landmark_name: str,
        axis: int
    ) -> float:
        """Calculate variance of a landmark position along an axis."""
        values = []
        for lm in landmarks_sequence:
            if lm and lm.get(landmark_name):
                if lm[landmark_name][2] > 0.3:  # visibility check
                    values.append(lm[landmark_name][axis])
        return np.var(values) if len(values) > 5 else 0.0


class GaitCycleDetector:
    """
    Manages heel strike events and builds gait cycles from pose data.

    Heel strikes are marked manually by the user. This class provides
    methods for managing events and building gait cycles from them.
    """

    def __init__(self, view_type: str, fps: float):
        """
        Initialize the detector.

        Args:
            view_type: 'side' or 'front'
            fps: Video frames per second
        """
        self.view_type = view_type
        self.fps = fps

        # Cycle timing constraints
        self.config = {
            'min_cycle_duration': 0.6,  # seconds
            'max_cycle_duration': 2.0,  # seconds
        }

        # Storage for events
        self._events: List[HeelStrikeEvent] = []

    def build_cycles(
        self,
        events: List[HeelStrikeEvent]
    ) -> List[GaitCycle]:
        """
        Build standard gait cycles from heel strike events.

        A standard gait cycle is defined from right heel strike to the next
        right heel strike. Each cycle must contain a left heel strike to be
        included (for bilateral comparison).

        Args:
            events: List of heel strike events (both sides)

        Returns:
            List of GaitCycle objects (only cycles with both R and L heel strikes)
        """
        # Separate by side
        left_events = sorted([e for e in events if e.side == 'left'],
                            key=lambda x: x.frame_idx)
        right_events = sorted([e for e in events if e.side == 'right'],
                             key=lambda x: x.frame_idx)

        cycles = []
        cycle_id = 0

        # Build cycles using right heel strikes as boundaries
        for i in range(len(right_events) - 1):
            start = right_events[i]
            end = right_events[i + 1]

            # Check duration is within acceptable range
            duration = end.time_seconds - start.time_seconds
            if not (self.config['min_cycle_duration'] <= duration <= self.config['max_cycle_duration']):
                continue

            # Find left heel strike(s) within this cycle
            # Look for left HS that occurs after start and before end
            left_in_cycle = [
                e for e in left_events
                if start.frame_idx < e.frame_idx < end.frame_idx
            ]

            # Require at least one left heel strike within the cycle
            if not left_in_cycle:
                continue

            # Use the first left heel strike if multiple exist
            # (multiple would indicate an unusually long cycle or detection issues)
            contralateral = left_in_cycle[0]

            cycles.append(GaitCycle(
                start_event=start,
                end_event=end,
                contralateral_event=contralateral,
                cycle_id=cycle_id,
            ))
            cycle_id += 1

        return cycles

    # Manual event management methods

    def add_manual_event(
        self,
        frame_idx: int,
        side: str,
        fps: float = None
    ) -> HeelStrikeEvent:
        """
        Add a manually marked heel strike event.

        Args:
            frame_idx: Frame index of the heel strike
            side: 'left' or 'right'
            fps: Frames per second (uses self.fps if not provided)

        Returns:
            The created HeelStrikeEvent
        """
        if fps is None:
            fps = self.fps

        event = HeelStrikeEvent(
            frame_idx=frame_idx,
            time_seconds=frame_idx / fps,
            side=side,
            confidence=1.0,  # Manual events have full confidence
            is_manual=True
        )
        self._events.append(event)
        self._events.sort(key=lambda e: e.frame_idx)
        return event

    def remove_event(self, event: HeelStrikeEvent) -> bool:
        """
        Remove a heel strike event.

        Returns:
            True if event was found and removed
        """
        try:
            self._events.remove(event)
            return True
        except ValueError:
            return False

    def modify_event_frame(
        self,
        event: HeelStrikeEvent,
        new_frame_idx: int
    ) -> HeelStrikeEvent:
        """
        Modify the frame index of an existing event.

        Args:
            event: The event to modify
            new_frame_idx: New frame index

        Returns:
            The modified event
        """
        event.frame_idx = new_frame_idx
        event.time_seconds = new_frame_idx / self.fps
        event.is_manual = True  # Mark as manually adjusted
        self._events.sort(key=lambda e: e.frame_idx)
        return event

    def get_events(self) -> List[HeelStrikeEvent]:
        """Get all currently tracked events."""
        return self._events.copy()

    def set_events(self, events: List[HeelStrikeEvent]):
        """Replace all events with a new list."""
        self._events = sorted(events, key=lambda e: e.frame_idx)

    def clear_events(self):
        """Clear all events."""
        self._events.clear()


@dataclass
class BatchProcessingResults:
    """Container for batch processing output."""
    video_path: str
    total_frames: int
    fps: float
    view_type: str
    view_type_confidence: float

    frame_data: Dict[int, FramePoseData]
    heel_strike_events: List[HeelStrikeEvent]
    gait_cycles: List[GaitCycle]

    processing_time_seconds: float
    frames_with_pose: int
    frames_without_pose: int


def batch_process_video(
    video_processor,  # VideoProcessor instance
    pose_detector,    # PoseDetector instance
    view_type: str,
    com_calculator = None,  # CenterOfMassCalculator instance
    progress_callback: Callable[[int, int], None] = None
) -> BatchProcessingResults:
    """
    Process entire video to extract pose data for gait analysis.

    This function processes all video frames to extract pose landmarks,
    joint angles, and center of mass data. Heel strikes must be marked
    manually by the user after processing.

    Args:
        video_processor: VideoProcessor instance
        pose_detector: PoseDetector instance
        view_type: 'side' or 'front'
        com_calculator: Optional CenterOfMassCalculator instance
        progress_callback: Optional callback(current_frame, total_frames)

    Returns:
        BatchProcessingResults with pose data (heel strikes and cycles empty,
        to be filled in via manual marking)
    """
    import time
    start_time = time.time()

    frame_data = {}
    frames_with_pose = 0
    frames_without_pose = 0

    # Process all frames
    for frame_idx in range(video_processor.total_frames):
        if progress_callback:
            progress_callback(frame_idx, video_processor.total_frames)

        frame = video_processor.get_frame(frame_idx)
        landmarks = pose_detector.detect(frame)

        if landmarks:
            # Get all angles including postural angles (view-type dependent)
            angles = pose_detector.get_all_angles(landmarks, view_type)
            com = None
            if com_calculator:
                com = com_calculator.calculate_com(landmarks)

            frame_data[frame_idx] = FramePoseData(
                frame_idx=frame_idx,
                timestamp=frame_idx / video_processor.fps,
                landmarks=landmarks,
                joint_angles=angles,
                center_of_mass=com
            )
            frames_with_pose += 1
        else:
            frames_without_pose += 1

    # Heel strikes and cycles are empty - user will mark them manually
    return BatchProcessingResults(
        video_path=video_processor.video_path,
        total_frames=video_processor.total_frames,
        fps=video_processor.fps,
        view_type=view_type,
        view_type_confidence=0.0,
        frame_data=frame_data,
        heel_strike_events=[],  # Manual marking required
        gait_cycles=[],         # Built from manual heel strikes
        processing_time_seconds=time.time() - start_time,
        frames_with_pose=frames_with_pose,
        frames_without_pose=frames_without_pose
    )
