"""
Gait cycle normalization module.

Normalizes gait cycle data to a standard 100-point format (0-100% of gait cycle)
for comparison across cycles and between individuals.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import interp1d

from gait_cycle import GaitCycle, FramePoseData


# Standard joint angle names in consistent order
# Includes both limb joint angles and postural/axial angles
ANGLE_NAMES = [
    # Limb joint angles (bilateral)
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    # Postural angles - sagittal plane (side view)
    'cervical_flexion',      # Forward head posture
    'thoracic_inclination',  # Upper trunk forward lean
    'trunk_inclination',     # Overall trunk forward lean
    # Postural angles - frontal plane (front view)
    'shoulder_tilt',         # Shoulder line from horizontal
    'hip_tilt',              # Hip/pelvis line from horizontal
    'trunk_lateral_lean',    # Trunk lateral deviation from vertical
    # Arm swing - lateral view only (near-side arm)
    'arm_swing_angle',       # Shoulder-wrist angle relative to trunk
]

# Standard landmark names (MediaPipe 33-point model)
LANDMARK_NAMES = [
    'nose',
    'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky', 'right_pinky',
    'left_index', 'right_index',
    'left_thumb', 'right_thumb',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]

NUM_NORMALIZED_POINTS = 100


@dataclass
class NormalizedGaitData:
    """
    Container for normalized gait cycle data.

    Each cycle is a standard gait cycle defined from right heel strike (0%)
    to the next right heel strike (100%). The contralateral (left) heel strike
    occurs within the cycle, typically around 50%.

    All bilateral joint angles are captured at each of the 100 normalized time points,
    allowing direct comparison of left vs right at the same phase of the gait cycle.
    """
    # Required fields (no defaults) - must come first
    cycle_id: int
    normalized_time: np.ndarray  # shape (100,) - Normalized time points (0-100%)
    joint_angles: np.ndarray  # shape (100, n_angles) - All angles at 100 time points
    landmarks: np.ndarray  # shape (100, 33, 3) - 33 landmarks at 100 time points
    center_of_mass: np.ndarray  # shape (100, 2) - x, y at 100 time points
    original_frame_indices: np.ndarray  # shape (100,) - original frame indices

    # Optional fields with defaults
    angle_names: List[str] = field(default_factory=lambda: ANGLE_NAMES.copy())
    landmark_names: List[str] = field(default_factory=lambda: LANDMARK_NAMES.copy())
    duration_seconds: float = 0.0
    fps: float = 30.0
    start_frame: int = 0
    end_frame: int = 0
    contralateral_timing: float = 50.0  # Normalized timing (0-100%) of left HS
    contralateral_frame: int = 0        # Frame index of left heel strike

    # Arm swing summary metrics (lateral view only)
    arm_swing_amplitude: float = 0.0           # Max - min angle during cycle (degrees)
    arm_swing_peak_forward_velocity: float = 0.0   # Peak forward swing speed (deg/sec)
    arm_swing_peak_backward_velocity: float = 0.0  # Peak backward swing speed (deg/sec)

    def get_angle_series(self, angle_name: str) -> np.ndarray:
        """Get time series for a specific joint angle."""
        if angle_name not in self.angle_names:
            raise ValueError(f"Unknown angle: {angle_name}")
        idx = self.angle_names.index(angle_name)
        return self.joint_angles[:, idx]

    def get_landmark_series(self, landmark_name: str) -> np.ndarray:
        """Get time series for a specific landmark (x, y, visibility)."""
        if landmark_name not in self.landmark_names:
            raise ValueError(f"Unknown landmark: {landmark_name}")
        idx = self.landmark_names.index(landmark_name)
        return self.landmarks[:, idx, :]

    def get_left_side_angles(self) -> np.ndarray:
        """Get all left-side joint angle time series."""
        left_indices = [i for i, name in enumerate(self.angle_names) if name.startswith('left_')]
        return self.joint_angles[:, left_indices]

    def get_right_side_angles(self) -> np.ndarray:
        """Get all right-side joint angle time series."""
        right_indices = [i for i, name in enumerate(self.angle_names) if name.startswith('right_')]
        return self.joint_angles[:, right_indices]


class GaitNormalizer:
    """
    Normalizes gait cycle data to a standard 100-point format.

    Uses linear interpolation to resample all measurements to exactly
    100 time points (0%, 1%, 2%, ... 99%, 100% of the gait cycle).
    """

    def __init__(self, num_points: int = NUM_NORMALIZED_POINTS):
        """
        Initialize the normalizer.

        Args:
            num_points: Number of normalized time points (default 100)
        """
        self.num_points = num_points
        self.normalized_time = np.linspace(0, 100, num_points)

    def normalize_cycle(
        self,
        cycle: GaitCycle,
        frame_data: Dict[int, FramePoseData],
        fps: float = 30.0
    ) -> Optional[NormalizedGaitData]:
        """
        Normalize a single gait cycle to the standard format.

        Args:
            cycle: GaitCycle object defining the cycle boundaries
            frame_data: Dictionary mapping frame indices to FramePoseData
            fps: Video frames per second

        Returns:
            NormalizedGaitData object, or None if insufficient data
        """
        # Get frames within this cycle
        cycle_frames = [f for f in cycle.frame_indices if f in frame_data]

        if len(cycle_frames) < 5:  # Need minimum frames for interpolation
            return None

        # Calculate original time points as percentage of cycle
        start_frame = cycle.start_frame
        end_frame = cycle.end_frame
        cycle_duration = end_frame - start_frame

        if cycle_duration <= 0:
            return None

        original_times = np.array([
            100.0 * (f - start_frame) / cycle_duration
            for f in cycle_frames
        ])

        # Extract and normalize joint angles
        joint_angles = self._normalize_angles(cycle_frames, frame_data, original_times)

        # Extract and normalize landmarks
        landmarks = self._normalize_landmarks(cycle_frames, frame_data, original_times)

        # Extract and normalize center of mass
        center_of_mass = self._normalize_com(cycle_frames, frame_data, original_times)

        # Calculate original frame indices at normalized time points
        original_frame_indices = np.interp(
            self.normalized_time,
            original_times,
            cycle_frames
        ).astype(int)

        # Compute arm swing summary metrics if arm_swing_angle is available
        arm_swing_amplitude = 0.0
        arm_swing_peak_forward_velocity = 0.0
        arm_swing_peak_backward_velocity = 0.0

        if 'arm_swing_angle' in ANGLE_NAMES:
            arm_swing_idx = ANGLE_NAMES.index('arm_swing_angle')
            arm_swing_series = joint_angles[:, arm_swing_idx]

            # Only compute if we have valid data
            valid_mask = ~np.isnan(arm_swing_series)
            if np.sum(valid_mask) >= 10:
                valid_angles = arm_swing_series[valid_mask]

                # Amplitude: max - min
                arm_swing_amplitude = float(np.nanmax(valid_angles) - np.nanmin(valid_angles))

                # Velocity: compute angular velocity in degrees per second
                # Each normalized time point represents 1% of the cycle
                # So dt between points = duration_seconds / 100
                if cycle.duration_seconds > 0:
                    dt = cycle.duration_seconds / 100.0  # seconds per normalized point

                    # Compute velocity using central differences where possible
                    velocity = np.full_like(arm_swing_series, np.nan)
                    for i in range(1, len(arm_swing_series) - 1):
                        if not np.isnan(arm_swing_series[i-1]) and not np.isnan(arm_swing_series[i+1]):
                            velocity[i] = (arm_swing_series[i+1] - arm_swing_series[i-1]) / (2 * dt)

                    # Peak forward velocity (positive = forward swing)
                    valid_vel = velocity[~np.isnan(velocity)]
                    if len(valid_vel) > 0:
                        arm_swing_peak_forward_velocity = float(np.nanmax(valid_vel))
                        arm_swing_peak_backward_velocity = float(np.nanmin(valid_vel))

        return NormalizedGaitData(
            cycle_id=cycle.cycle_id,
            normalized_time=self.normalized_time.copy(),
            joint_angles=joint_angles,
            landmarks=landmarks,
            center_of_mass=center_of_mass,
            original_frame_indices=original_frame_indices,
            duration_seconds=cycle.duration_seconds,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            contralateral_timing=cycle.contralateral_timing,
            contralateral_frame=cycle.contralateral_frame,
            arm_swing_amplitude=arm_swing_amplitude,
            arm_swing_peak_forward_velocity=arm_swing_peak_forward_velocity,
            arm_swing_peak_backward_velocity=arm_swing_peak_backward_velocity,
        )

    def _normalize_angles(
        self,
        cycle_frames: List[int],
        frame_data: Dict[int, FramePoseData],
        original_times: np.ndarray
    ) -> np.ndarray:
        """Normalize joint angles to 100 time points."""
        num_angles = len(ANGLE_NAMES)
        result = np.full((self.num_points, num_angles), np.nan)

        for angle_idx, angle_name in enumerate(ANGLE_NAMES):
            # Extract angle values for this cycle
            values = []
            times = []

            for i, frame_idx in enumerate(cycle_frames):
                data = frame_data[frame_idx]
                if data.joint_angles and angle_name in data.joint_angles:
                    angle_val = data.joint_angles[angle_name]
                    if angle_val is not None:
                        values.append(angle_val)
                        times.append(original_times[i])

            if len(values) >= 3:  # Need at least 3 points for interpolation
                # Interpolate to normalized time points
                result[:, angle_idx] = self._interpolate_series(
                    np.array(times), np.array(values)
                )

        return result

    def _normalize_landmarks(
        self,
        cycle_frames: List[int],
        frame_data: Dict[int, FramePoseData],
        original_times: np.ndarray
    ) -> np.ndarray:
        """Normalize landmarks to 100 time points."""
        num_landmarks = len(LANDMARK_NAMES)
        result = np.full((self.num_points, num_landmarks, 3), np.nan)

        for lm_idx, lm_name in enumerate(LANDMARK_NAMES):
            # Extract landmark values
            x_values, y_values, vis_values = [], [], []
            times = []

            for i, frame_idx in enumerate(cycle_frames):
                data = frame_data[frame_idx]
                if data.landmarks and lm_name in data.landmarks:
                    lm = data.landmarks[lm_name]
                    if lm[2] > 0.1:  # Minimum visibility
                        x_values.append(lm[0])
                        y_values.append(lm[1])
                        vis_values.append(lm[2])
                        times.append(original_times[i])

            if len(times) >= 3:
                times_arr = np.array(times)
                result[:, lm_idx, 0] = self._interpolate_series(times_arr, np.array(x_values))
                result[:, lm_idx, 1] = self._interpolate_series(times_arr, np.array(y_values))
                result[:, lm_idx, 2] = self._interpolate_series(times_arr, np.array(vis_values))

        return result

    def _normalize_com(
        self,
        cycle_frames: List[int],
        frame_data: Dict[int, FramePoseData],
        original_times: np.ndarray
    ) -> np.ndarray:
        """Normalize center of mass to 100 time points."""
        result = np.full((self.num_points, 2), np.nan)

        x_values, y_values, times = [], [], []

        for i, frame_idx in enumerate(cycle_frames):
            data = frame_data[frame_idx]
            if data.center_of_mass is not None:
                x_values.append(data.center_of_mass[0])
                y_values.append(data.center_of_mass[1])
                times.append(original_times[i])

        if len(times) >= 3:
            times_arr = np.array(times)
            result[:, 0] = self._interpolate_series(times_arr, np.array(x_values))
            result[:, 1] = self._interpolate_series(times_arr, np.array(y_values))

        return result

    def _interpolate_series(
        self,
        original_times: np.ndarray,
        values: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate a time series to normalized time points.

        Uses linear interpolation with extrapolation handling.
        """
        if len(original_times) < 2:
            return np.full(self.num_points, np.nan)

        # Create interpolation function
        interp_func = interp1d(
            original_times,
            values,
            kind='linear',
            bounds_error=False,
            fill_value=(values[0], values[-1])  # Extrapolate with edge values
        )

        return interp_func(self.normalized_time)


def compute_cycle_statistics(
    normalized_cycles: List[NormalizedGaitData]
) -> Dict[str, np.ndarray]:
    """
    Compute statistics across multiple standard gait cycles.

    Args:
        normalized_cycles: List of normalized gait cycle data

    Returns:
        Dictionary with 'mean', 'std', 'min', 'max' arrays for:
        - joint_angles: (100, n_angles) for each statistic
        - center_of_mass: (100, 2) for each statistic
        - contralateral_timing statistics
    """
    if not normalized_cycles:
        return {}

    cycles = normalized_cycles

    # Stack angle arrays
    angles_stack = np.stack([c.joint_angles for c in cycles], axis=0)
    com_stack = np.stack([c.center_of_mass for c in cycles], axis=0)
    contra_timings = np.array([c.contralateral_timing for c in cycles])

    # Compute statistics (ignoring NaN)
    with np.errstate(all='ignore'):
        return {
            'joint_angles_mean': np.nanmean(angles_stack, axis=0),
            'joint_angles_std': np.nanstd(angles_stack, axis=0),
            'joint_angles_min': np.nanmin(angles_stack, axis=0),
            'joint_angles_max': np.nanmax(angles_stack, axis=0),
            'center_of_mass_mean': np.nanmean(com_stack, axis=0),
            'center_of_mass_std': np.nanstd(com_stack, axis=0),
            'center_of_mass_min': np.nanmin(com_stack, axis=0),
            'center_of_mass_max': np.nanmax(com_stack, axis=0),
            'contralateral_timing_mean': np.nanmean(contra_timings),
            'contralateral_timing_std': np.nanstd(contra_timings),
            'num_cycles': len(cycles),
            'cycle_ids': [c.cycle_id for c in cycles],
        }


def compare_contralateral(
    normalized_cycles: List[NormalizedGaitData],
    angle_name_base: str
) -> Dict[str, np.ndarray]:
    """
    Compare left vs right side for a specific angle type.

    Args:
        normalized_cycles: List of normalized gait cycle data
        angle_name_base: Base name without side prefix (e.g., 'knee', 'hip')

    Returns:
        Dictionary with comparison data
    """
    left_angle = f'left_{angle_name_base}'
    right_angle = f'right_{angle_name_base}'

    left_data = []
    right_data = []

    for cycle in normalized_cycles:
        try:
            left_series = cycle.get_angle_series(left_angle)
            right_series = cycle.get_angle_series(right_angle)

            if not np.all(np.isnan(left_series)):
                left_data.append(left_series)
            if not np.all(np.isnan(right_series)):
                right_data.append(right_series)
        except ValueError:
            continue

    result = {}

    if left_data:
        left_stack = np.stack(left_data, axis=0)
        result['left_mean'] = np.nanmean(left_stack, axis=0)
        result['left_std'] = np.nanstd(left_stack, axis=0)
        result['left_n'] = len(left_data)

    if right_data:
        right_stack = np.stack(right_data, axis=0)
        result['right_mean'] = np.nanmean(right_stack, axis=0)
        result['right_std'] = np.nanstd(right_stack, axis=0)
        result['right_n'] = len(right_data)

    if 'left_mean' in result and 'right_mean' in result:
        result['difference_mean'] = result['left_mean'] - result['right_mean']

    return result


def compare_angle_across_cycles(
    normalized_cycles: List[NormalizedGaitData],
    angle_name: str
) -> Dict[str, np.ndarray]:
    """
    Compare a specific angle across multiple gait cycles.

    Useful for assessing gait consistency/variability within a recording session.

    Args:
        normalized_cycles: List of normalized gait cycle data
        angle_name: Full angle name (e.g., 'left_knee', 'right_hip')

    Returns:
        Dictionary with individual cycle data and statistics
    """
    cycles = normalized_cycles

    if not cycles:
        return {}

    cycle_data = []
    cycle_ids = []

    for cycle in cycles:
        try:
            series = cycle.get_angle_series(angle_name)
            if not np.all(np.isnan(series)):
                cycle_data.append(series)
                cycle_ids.append(cycle.cycle_id)
        except ValueError:
            continue

    if not cycle_data:
        return {}

    data_stack = np.stack(cycle_data, axis=0)  # (n_cycles, 100)

    return {
        'individual_cycles': data_stack,
        'cycle_ids': cycle_ids,
        'mean': np.nanmean(data_stack, axis=0),
        'std': np.nanstd(data_stack, axis=0),
        'coefficient_of_variation': np.nanstd(data_stack, axis=0) / (np.abs(np.nanmean(data_stack, axis=0)) + 1e-6),
        'num_cycles': len(cycle_data),
    }
