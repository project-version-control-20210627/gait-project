"""
Machine Learning feature extraction module for gait analysis.

Transforms normalized gait cycle data into ML-ready feature matrices suitable for:
- Clustering and pattern recognition
- Anomaly detection
- Longitudinal change detection
- Classification tasks

The feature matrix format is designed for direct use with scikit-learn, PyTorch,
TensorFlow, and other ML frameworks.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from gait_normalizer import NormalizedGaitData, ANGLE_NAMES


# Feature categories for organization
JOINT_ANGLE_FEATURES = [
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

POSTURAL_ANGLE_FEATURES_SIDE = [
    'cervical_flexion',
    'thoracic_inclination',
    'trunk_inclination',
]

POSTURAL_ANGLE_FEATURES_FRONT = [
    'shoulder_tilt',
    'hip_tilt',
    'trunk_lateral_lean',
]

COM_FEATURES = ['com_x', 'com_y']

# Number of time points in normalized gait cycle
NUM_TIME_POINTS = 100


@dataclass
class GaitFeatureVector:
    """
    Feature vector for a single standard gait cycle.

    Contains the flattened time-series data plus derived summary features.
    Each cycle is a standard gait cycle from right heel strike to right heel strike,
    with contralateral (left) heel strike occurring within the cycle.
    """
    cycle_id: int

    # Flattened time-series features: shape (n_time_series_features,)
    # Layout: [angle1_t0, angle1_t1, ..., angle1_t99, angle2_t0, ..., com_y_t99]
    time_series_features: np.ndarray
    time_series_feature_names: List[str]

    # Derived summary features: shape (n_summary_features,)
    summary_features: np.ndarray
    summary_feature_names: List[str]

    # Combined feature vector for ML: shape (n_total_features,)
    @property
    def feature_vector(self) -> np.ndarray:
        """Get combined feature vector (time-series + summary)."""
        return np.concatenate([self.time_series_features, self.summary_features])

    @property
    def feature_names(self) -> List[str]:
        """Get all feature names in order."""
        return self.time_series_feature_names + self.summary_feature_names

    # Metadata
    duration_seconds: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    contralateral_timing: float = 50.0  # When left HS occurs (0-100%)


@dataclass
class MLGaitDataset:
    """
    Complete ML-ready dataset from gait analysis.

    All cycles are standard gait cycles defined from right heel strike to right
    heel strike, with contralateral (left) heel strike occurring within each cycle.

    Designed for direct use with ML frameworks:
    - feature_matrix: (n_cycles, n_features) for sklearn, etc.
    - Can be easily converted to PyTorch tensors or TensorFlow datasets
    """
    # Main feature matrix: (n_cycles, n_features)
    feature_matrix: np.ndarray

    # Feature names: length n_features
    feature_names: List[str]

    # Time-series only matrix (without summary features)
    time_series_matrix: np.ndarray
    time_series_feature_names: List[str]

    # Summary features only matrix
    summary_matrix: np.ndarray
    summary_feature_names: List[str]

    # Labels and metadata per cycle: length n_cycles
    cycle_ids: np.ndarray
    contralateral_timings: np.ndarray  # When left HS occurs (0-100%) for each cycle
    durations: np.ndarray

    # Dataset metadata
    n_cycles: int = 0
    n_features: int = 0
    n_time_series_features: int = 0
    n_summary_features: int = 0
    view_type: str = 'side'

    # Optional subject/session info for longitudinal studies
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    recording_date: Optional[str] = None

    # Normalization parameters (for applying to new data)
    feature_means: Optional[np.ndarray] = None
    feature_stds: Optional[np.ndarray] = None

    def get_normalized_features(self) -> np.ndarray:
        """
        Get z-score normalized feature matrix.

        Returns:
            Normalized feature matrix with zero mean and unit variance per feature.
        """
        if self.feature_means is None or self.feature_stds is None:
            self.compute_normalization_params()

        # Avoid division by zero for constant features
        stds = np.where(self.feature_stds > 1e-10, self.feature_stds, 1.0)
        return (self.feature_matrix - self.feature_means) / stds

    def compute_normalization_params(self):
        """Compute and store normalization parameters."""
        self.feature_means = np.nanmean(self.feature_matrix, axis=0)
        self.feature_stds = np.nanstd(self.feature_matrix, axis=0)

    def get_asymmetric_cycles(self, threshold: float = 5.0) -> np.ndarray:
        """Get feature matrix for cycles with asymmetric timing (>threshold% from 50%)."""
        mask = np.abs(self.contralateral_timings - 50.0) > threshold
        return self.feature_matrix[mask]

    def get_symmetric_cycles(self, threshold: float = 5.0) -> np.ndarray:
        """Get feature matrix for cycles with symmetric timing (within threshold% of 50%)."""
        mask = np.abs(self.contralateral_timings - 50.0) <= threshold
        return self.feature_matrix[mask]

    def get_cycle_by_id(self, cycle_id: int) -> Optional[np.ndarray]:
        """Get feature vector for a specific cycle."""
        mask = self.cycle_ids == cycle_id
        if np.any(mask):
            return self.feature_matrix[mask][0]
        return None

    def get_step_asymmetry_stats(self) -> Dict[str, float]:
        """Get statistics on step time asymmetry across cycles."""
        asymmetries = self.contralateral_timings - 50.0
        return {
            'mean_asymmetry': np.nanmean(asymmetries),
            'std_asymmetry': np.nanstd(asymmetries),
            'max_asymmetry': np.nanmax(np.abs(asymmetries)),
            'n_asymmetric': np.sum(np.abs(asymmetries) > 5.0),
            'n_total': len(asymmetries),
        }


class GaitFeatureExtractor:
    """
    Extracts ML-ready features from normalized gait cycle data.

    Features include:
    1. Time-series features: All angles and CoM at 100 time points (flattened)
    2. Summary features: ROM, mean, std, symmetry indices, etc.
    """

    def __init__(self, view_type: str = 'side'):
        """
        Initialize the feature extractor.

        Args:
            view_type: 'side' or 'front' - determines which postural angles to use
        """
        self.view_type = view_type
        self._setup_feature_names()

    def _setup_feature_names(self):
        """Set up feature name lists based on view type."""
        # Determine which angles to use based on view type
        if self.view_type == 'side':
            self.angle_names = JOINT_ANGLE_FEATURES + POSTURAL_ANGLE_FEATURES_SIDE
        else:
            self.angle_names = JOINT_ANGLE_FEATURES + POSTURAL_ANGLE_FEATURES_FRONT

        # Time-series feature names: angle_name_tXX for each time point
        self.ts_feature_names = []
        for angle in self.angle_names:
            for t in range(NUM_TIME_POINTS):
                self.ts_feature_names.append(f"{angle}_t{t:02d}")

        # Add CoM time-series features
        for com in COM_FEATURES:
            for t in range(NUM_TIME_POINTS):
                self.ts_feature_names.append(f"{com}_t{t:02d}")

        # Summary feature names
        self.summary_feature_names = self._get_summary_feature_names()

    def _get_summary_feature_names(self) -> List[str]:
        """Get list of summary feature names."""
        names = []

        # Range of motion for each angle
        for angle in self.angle_names:
            names.append(f"{angle}_rom")
            names.append(f"{angle}_mean")
            names.append(f"{angle}_std")
            names.append(f"{angle}_min")
            names.append(f"{angle}_max")

        # CoM summary features
        for com in COM_FEATURES:
            names.append(f"{com}_range")
            names.append(f"{com}_mean")
            names.append(f"{com}_std")

        # Symmetry indices (comparing left vs right for bilateral angles)
        bilateral_bases = ['shoulder', 'elbow', 'hip', 'knee', 'ankle']
        for base in bilateral_bases:
            names.append(f"{base}_symmetry_rom")    # ROM difference
            names.append(f"{base}_symmetry_mean")   # Mean difference
            names.append(f"{base}_symmetry_phase")  # Phase shift correlation

        # Gait-specific derived features
        names.extend([
            'cadence_proxy',           # Inverse of duration
            'step_length_proxy',       # CoM x range
            'vertical_oscillation',    # CoM y range
            'stability_index',         # CoM trajectory smoothness
        ])

        return names

    def extract_features(
        self,
        normalized_cycles: List[NormalizedGaitData]
    ) -> MLGaitDataset:
        """
        Extract ML features from a list of normalized gait cycles.

        Args:
            normalized_cycles: List of NormalizedGaitData objects

        Returns:
            MLGaitDataset ready for machine learning
        """
        if not normalized_cycles:
            return self._create_empty_dataset()

        # Extract features for each cycle
        feature_vectors = []
        for cycle in normalized_cycles:
            fv = self._extract_cycle_features(cycle)
            feature_vectors.append(fv)

        # Build the dataset
        return self._build_dataset(feature_vectors)

    def _extract_cycle_features(
        self,
        cycle: NormalizedGaitData
    ) -> GaitFeatureVector:
        """Extract features from a single normalized cycle."""

        # Extract time-series features (flattened)
        ts_features = self._extract_time_series_features(cycle)

        # Extract summary features
        summary_features = self._extract_summary_features(cycle)

        return GaitFeatureVector(
            cycle_id=cycle.cycle_id,
            time_series_features=ts_features,
            time_series_feature_names=self.ts_feature_names.copy(),
            summary_features=summary_features,
            summary_feature_names=self.summary_feature_names.copy(),
            duration_seconds=cycle.duration_seconds,
            start_frame=cycle.start_frame,
            end_frame=cycle.end_frame,
            contralateral_timing=cycle.contralateral_timing,
        )

    def _extract_time_series_features(
        self,
        cycle: NormalizedGaitData
    ) -> np.ndarray:
        """
        Extract flattened time-series features.

        Layout: [angle1_t0, angle1_t1, ..., angle1_t99, angle2_t0, ..., com_y_t99]
        """
        features = []

        # Extract angle time series
        for angle_name in self.angle_names:
            if angle_name in cycle.angle_names:
                idx = cycle.angle_names.index(angle_name)
                angle_series = cycle.joint_angles[:, idx]
            else:
                # Angle not available (e.g., wrong view type)
                angle_series = np.full(NUM_TIME_POINTS, np.nan)
            features.extend(angle_series)

        # Extract CoM time series
        features.extend(cycle.center_of_mass[:, 0])  # com_x
        features.extend(cycle.center_of_mass[:, 1])  # com_y

        return np.array(features)

    def _extract_summary_features(
        self,
        cycle: NormalizedGaitData
    ) -> np.ndarray:
        """
        Extract summary/derived features.

        Includes ROM, statistics, symmetry indices, and gait-specific metrics.
        """
        features = []

        # Angle statistics
        for angle_name in self.angle_names:
            if angle_name in cycle.angle_names:
                idx = cycle.angle_names.index(angle_name)
                series = cycle.joint_angles[:, idx]
            else:
                series = np.full(NUM_TIME_POINTS, np.nan)

            valid = ~np.isnan(series)
            if np.sum(valid) > 0:
                features.append(np.nanmax(series) - np.nanmin(series))  # ROM
                features.append(np.nanmean(series))
                features.append(np.nanstd(series))
                features.append(np.nanmin(series))
                features.append(np.nanmax(series))
            else:
                features.extend([np.nan] * 5)

        # CoM statistics
        for i, com_name in enumerate(COM_FEATURES):
            series = cycle.center_of_mass[:, i]
            valid = ~np.isnan(series)
            if np.sum(valid) > 0:
                features.append(np.nanmax(series) - np.nanmin(series))  # range
                features.append(np.nanmean(series))
                features.append(np.nanstd(series))
            else:
                features.extend([np.nan] * 3)

        # Symmetry indices for bilateral angles
        bilateral_bases = ['shoulder', 'elbow', 'hip', 'knee', 'ankle']
        for base in bilateral_bases:
            left_name = f'left_{base}'
            right_name = f'right_{base}'

            left_series = self._get_angle_series(cycle, left_name)
            right_series = self._get_angle_series(cycle, right_name)

            if left_series is not None and right_series is not None:
                # ROM symmetry (difference in range of motion)
                left_rom = np.nanmax(left_series) - np.nanmin(left_series)
                right_rom = np.nanmax(right_series) - np.nanmin(right_series)
                features.append(left_rom - right_rom)

                # Mean symmetry
                features.append(np.nanmean(left_series) - np.nanmean(right_series))

                # Phase symmetry (cross-correlation at lag 0)
                # Higher values indicate more similar patterns
                valid_mask = ~(np.isnan(left_series) | np.isnan(right_series))
                if np.sum(valid_mask) > 10:
                    left_norm = (left_series[valid_mask] - np.mean(left_series[valid_mask]))
                    right_norm = (right_series[valid_mask] - np.mean(right_series[valid_mask]))
                    denom = np.std(left_norm) * np.std(right_norm) * len(left_norm)
                    if denom > 1e-10:
                        phase_corr = np.sum(left_norm * right_norm) / denom
                    else:
                        phase_corr = np.nan
                else:
                    phase_corr = np.nan
                features.append(phase_corr)
            else:
                features.extend([np.nan] * 3)

        # Gait-specific derived features
        features.append(1.0 / cycle.duration_seconds if cycle.duration_seconds > 0 else np.nan)  # cadence_proxy

        com_x = cycle.center_of_mass[:, 0]
        com_y = cycle.center_of_mass[:, 1]

        valid_x = ~np.isnan(com_x)
        valid_y = ~np.isnan(com_y)

        # Step length proxy (CoM x range)
        if np.sum(valid_x) > 0:
            features.append(np.nanmax(com_x) - np.nanmin(com_x))
        else:
            features.append(np.nan)

        # Vertical oscillation (CoM y range)
        if np.sum(valid_y) > 0:
            features.append(np.nanmax(com_y) - np.nanmin(com_y))
        else:
            features.append(np.nan)

        # Stability index (smoothness of CoM trajectory)
        # Lower values = smoother = more stable
        if np.sum(valid_x) > 5 and np.sum(valid_y) > 5:
            # Use second derivative (acceleration) variance as smoothness measure
            com_x_interp = np.interp(np.arange(100), np.where(valid_x)[0], com_x[valid_x])
            com_y_interp = np.interp(np.arange(100), np.where(valid_y)[0], com_y[valid_y])
            accel_x = np.diff(np.diff(com_x_interp))
            accel_y = np.diff(np.diff(com_y_interp))
            stability = np.sqrt(np.var(accel_x) + np.var(accel_y))
            features.append(stability)
        else:
            features.append(np.nan)

        return np.array(features)

    def _get_angle_series(
        self,
        cycle: NormalizedGaitData,
        angle_name: str
    ) -> Optional[np.ndarray]:
        """Get angle time series by name, or None if not available."""
        if angle_name in cycle.angle_names:
            idx = cycle.angle_names.index(angle_name)
            series = cycle.joint_angles[:, idx]
            if not np.all(np.isnan(series)):
                return series
        return None

    def _build_dataset(
        self,
        feature_vectors: List[GaitFeatureVector]
    ) -> MLGaitDataset:
        """Build MLGaitDataset from list of feature vectors."""
        n_cycles = len(feature_vectors)

        # Stack feature matrices
        ts_matrix = np.vstack([fv.time_series_features for fv in feature_vectors])
        summary_matrix = np.vstack([fv.summary_features for fv in feature_vectors])
        full_matrix = np.hstack([ts_matrix, summary_matrix])

        # Collect metadata
        cycle_ids = np.array([fv.cycle_id for fv in feature_vectors])
        contralateral_timings = np.array([fv.contralateral_timing for fv in feature_vectors])
        durations = np.array([fv.duration_seconds for fv in feature_vectors])

        # Feature names
        all_feature_names = feature_vectors[0].feature_names

        return MLGaitDataset(
            feature_matrix=full_matrix,
            feature_names=all_feature_names,
            time_series_matrix=ts_matrix,
            time_series_feature_names=feature_vectors[0].time_series_feature_names,
            summary_matrix=summary_matrix,
            summary_feature_names=feature_vectors[0].summary_feature_names,
            cycle_ids=cycle_ids,
            contralateral_timings=contralateral_timings,
            durations=durations,
            n_cycles=n_cycles,
            n_features=full_matrix.shape[1],
            n_time_series_features=ts_matrix.shape[1],
            n_summary_features=summary_matrix.shape[1],
            view_type=self.view_type,
        )

    def _create_empty_dataset(self) -> MLGaitDataset:
        """Create an empty dataset."""
        return MLGaitDataset(
            feature_matrix=np.array([]).reshape(0, 0),
            feature_names=[],
            time_series_matrix=np.array([]).reshape(0, 0),
            time_series_feature_names=[],
            summary_matrix=np.array([]).reshape(0, 0),
            summary_feature_names=[],
            cycle_ids=np.array([]),
            contralateral_timings=np.array([]),
            durations=np.array([]),
            n_cycles=0,
            n_features=0,
            n_time_series_features=0,
            n_summary_features=0,
            view_type=self.view_type,
        )


def compute_gait_signature(
    dataset: MLGaitDataset,
    symmetric_only: bool = False,
    asymmetry_threshold: float = 5.0
) -> Dict[str, np.ndarray]:
    """
    Compute a "gait signature" baseline from multiple standard gait cycles.

    The signature consists of the mean and standard deviation of each feature
    across cycles. This can be used to:
    - Establish an individual's baseline gait pattern
    - Detect deviations from baseline in new recordings
    - Compare gait patterns between individuals

    Args:
        dataset: MLGaitDataset with extracted features
        symmetric_only: If True, only include cycles with symmetric timing
        asymmetry_threshold: Threshold for symmetric timing (deviation from 50%)

    Returns:
        Dictionary with 'mean', 'std', 'median', 'iqr' arrays and asymmetry stats
    """
    if symmetric_only:
        features = dataset.get_symmetric_cycles(asymmetry_threshold)
    else:
        features = dataset.feature_matrix

    if len(features) == 0:
        return {}

    # Compute asymmetry statistics
    asymmetry_stats = dataset.get_step_asymmetry_stats()

    return {
        'mean': np.nanmean(features, axis=0),
        'std': np.nanstd(features, axis=0),
        'median': np.nanmedian(features, axis=0),
        'iqr': np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0),
        'n_cycles': len(features),
        'feature_names': dataset.feature_names,
        'mean_contralateral_timing': np.nanmean(dataset.contralateral_timings),
        'std_contralateral_timing': np.nanstd(dataset.contralateral_timings),
        'step_asymmetry_stats': asymmetry_stats,
    }


def compute_deviation_scores(
    new_cycle_features: np.ndarray,
    baseline_signature: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute z-score deviations from a baseline signature.

    Useful for detecting changes over time or identifying abnormal gait patterns.

    Args:
        new_cycle_features: Feature vector for a new cycle (1D array)
        baseline_signature: Signature dict from compute_gait_signature()

    Returns:
        Array of z-scores for each feature
    """
    mean = baseline_signature['mean']
    std = baseline_signature['std']

    # Avoid division by zero
    std_safe = np.where(std > 1e-10, std, 1.0)

    return (new_cycle_features - mean) / std_safe


def compare_sessions(
    session1: MLGaitDataset,
    session2: MLGaitDataset,
    features_to_compare: List[str] = None
) -> Dict[str, any]:
    """
    Compare gait features between two recording sessions.

    Useful for longitudinal tracking and detecting changes over time.

    Args:
        session1: First session's dataset (e.g., baseline)
        session2: Second session's dataset (e.g., follow-up)
        features_to_compare: Optional list of specific features to compare

    Returns:
        Dictionary with comparison statistics
    """
    if features_to_compare is None:
        features_to_compare = session1.feature_names

    results = {
        'session1_n_cycles': session1.n_cycles,
        'session2_n_cycles': session2.n_cycles,
        'feature_comparisons': {},
    }

    for feat_name in features_to_compare:
        if feat_name not in session1.feature_names or feat_name not in session2.feature_names:
            continue

        idx1 = session1.feature_names.index(feat_name)
        idx2 = session2.feature_names.index(feat_name)

        vals1 = session1.feature_matrix[:, idx1]
        vals2 = session2.feature_matrix[:, idx2]

        # Remove NaN values
        vals1 = vals1[~np.isnan(vals1)]
        vals2 = vals2[~np.isnan(vals2)]

        if len(vals1) > 0 and len(vals2) > 0:
            # Compute comparison statistics
            mean_diff = np.mean(vals2) - np.mean(vals1)
            percent_change = (mean_diff / np.abs(np.mean(vals1))) * 100 if np.mean(vals1) != 0 else np.nan

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(vals1) + np.var(vals2)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 1e-10 else np.nan

            results['feature_comparisons'][feat_name] = {
                'session1_mean': np.mean(vals1),
                'session1_std': np.std(vals1),
                'session2_mean': np.mean(vals2),
                'session2_std': np.std(vals2),
                'mean_difference': mean_diff,
                'percent_change': percent_change,
                'cohens_d': cohens_d,
            }

    return results
