"""
Gait data export module.

Provides functionality to export gait cycle data in various formats
including NumPy arrays, pickle files, CSV, and ML-ready feature matrices.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import pickle
import json
import os
from datetime import datetime

from gait_cycle import HeelStrikeEvent, GaitCycle, BatchProcessingResults, SubjectMetadata, DualViewSession
from gait_normalizer import NormalizedGaitData, ANGLE_NAMES, LANDMARK_NAMES
from gait_ml_features import (
    GaitFeatureExtractor, MLGaitDataset, compute_gait_signature
)


@dataclass
class GaitExportMetadata:
    """Metadata for gait export package."""
    video_path: str
    video_filename: str
    export_timestamp: str
    view_type: str
    fps: float
    total_frames: int
    num_cycles: int
    mean_contralateral_timing: float  # Mean timing of left HS within cycles (0-100%)
    std_contralateral_timing: float   # Std of timing
    processing_time_seconds: float
    frames_with_pose: int
    frames_without_pose: int


@dataclass
class GaitExportPackage:
    """
    Complete export package for gait analysis data.

    All cycles are standard gait cycles defined from right heel strike to
    right heel strike, with contralateral (left) heel strike tracked within.
    """
    # Metadata
    metadata: GaitExportMetadata

    # Raw events (serialized)
    heel_strikes: List[Dict]

    # Cycle info
    cycle_info: List[Dict]

    # Normalized data arrays
    # Shape: (num_cycles, 100, num_features)
    normalized_joint_angles: np.ndarray      # (n_cycles, 100, n_angles)
    normalized_landmarks: np.ndarray          # (n_cycles, 100, 33, 3)
    normalized_center_of_mass: np.ndarray     # (n_cycles, 100, 2)

    # Contralateral timing per cycle
    contralateral_timings: np.ndarray  # (n_cycles,) - When left HS occurs (0-100%)

    # Labels
    angle_names: List[str]
    landmark_names: List[str]


@dataclass
class DualViewExportMetadata:
    """Metadata for dual-view gait export package."""
    # Subject metadata
    subject_id: str
    date: str
    walking_condition: str

    # Export info
    export_timestamp: str

    # Sagittal view info
    sagittal_video_path: str
    sagittal_video_filename: str
    sagittal_camera_side: str  # 'left' or 'right'
    sagittal_fps: float
    sagittal_total_frames: int
    sagittal_num_cycles: int
    sagittal_mean_contralateral_timing: float
    sagittal_mean_duration: float
    sagittal_mean_asymmetry: float
    sagittal_mean_arm_swing: float

    # Frontal view info
    frontal_video_path: str
    frontal_video_filename: str
    frontal_fps: float
    frontal_total_frames: int
    frontal_num_cycles: int


@dataclass
class DualViewExportPackage:
    """
    Complete export package for dual-view gait analysis data.

    Contains both sagittal and frontal view data along with subject metadata.
    Sagittal view includes duration, asymmetry, and arm swing metrics.
    Frontal view includes normalized joint angles only.
    """
    # Metadata
    metadata: DualViewExportMetadata

    # Sagittal view data
    sagittal_heel_strikes: List[Dict]
    sagittal_cycle_info: List[Dict]  # Includes duration, asymmetry, arm swing
    sagittal_normalized_joint_angles: np.ndarray  # (n_cycles, 100, n_angles)
    sagittal_normalized_landmarks: np.ndarray     # (n_cycles, 100, 33, 3)
    sagittal_normalized_center_of_mass: np.ndarray  # (n_cycles, 100, 2)
    sagittal_contralateral_timings: np.ndarray    # (n_cycles,)

    # Frontal view data
    frontal_heel_strikes: List[Dict]
    frontal_cycle_info: List[Dict]  # Basic info only
    frontal_normalized_joint_angles: np.ndarray   # (n_cycles, 100, n_angles)
    frontal_normalized_landmarks: np.ndarray      # (n_cycles, 100, 33, 3)
    frontal_normalized_center_of_mass: np.ndarray # (n_cycles, 100, 2)
    frontal_contralateral_timings: np.ndarray     # (n_cycles,)

    # Labels (same for both views)
    angle_names: List[str]
    landmark_names: List[str]


class GaitDataExporter:
    """Handles export of gait analysis data in various formats."""

    def create_export_package(
        self,
        batch_results: BatchProcessingResults,
        normalized_cycles: List[NormalizedGaitData]
    ) -> GaitExportPackage:
        """
        Create a complete export package from processing results.

        All cycles are standard gait cycles (R HS to R HS) with contralateral
        (left) heel strike tracked within each cycle.

        Args:
            batch_results: Results from batch video processing
            normalized_cycles: List of normalized gait cycle data

        Returns:
            GaitExportPackage ready for export
        """
        # Calculate contralateral timing statistics
        contra_timings = np.array([c.contralateral_timing for c in normalized_cycles]) if normalized_cycles else np.array([])
        mean_contra = float(np.nanmean(contra_timings)) if len(contra_timings) > 0 else 50.0
        std_contra = float(np.nanstd(contra_timings)) if len(contra_timings) > 0 else 0.0

        # Create metadata
        metadata = GaitExportMetadata(
            video_path=batch_results.video_path,
            video_filename=os.path.basename(batch_results.video_path),
            export_timestamp=datetime.now().isoformat(),
            view_type=batch_results.view_type,
            fps=batch_results.fps,
            total_frames=batch_results.total_frames,
            num_cycles=len(normalized_cycles),
            mean_contralateral_timing=mean_contra,
            std_contralateral_timing=std_contra,
            processing_time_seconds=batch_results.processing_time_seconds,
            frames_with_pose=batch_results.frames_with_pose,
            frames_without_pose=batch_results.frames_without_pose
        )

        # Serialize heel strike events
        heel_strikes = [
            {
                'frame_idx': e.frame_idx,
                'time_seconds': e.time_seconds,
                'side': e.side,
                'confidence': e.confidence,
                'is_manual': e.is_manual
            }
            for e in batch_results.heel_strike_events
        ]

        # Serialize cycle info
        cycle_info = [
            {
                'cycle_id': c.cycle_id,
                'start_frame': c.start_frame,
                'end_frame': c.end_frame,
                'duration_seconds': c.duration_seconds,
                'contralateral_timing': c.contralateral_timing,
                'contralateral_frame': c.contralateral_frame,
                'arm_swing_amplitude': c.arm_swing_amplitude,
                'arm_swing_peak_forward_velocity': c.arm_swing_peak_forward_velocity,
                'arm_swing_peak_backward_velocity': c.arm_swing_peak_backward_velocity,
            }
            for c in normalized_cycles
        ]

        # Stack normalized arrays
        if normalized_cycles:
            normalized_angles = np.stack([c.joint_angles for c in normalized_cycles], axis=0)
            normalized_landmarks = np.stack([c.landmarks for c in normalized_cycles], axis=0)
            normalized_com = np.stack([c.center_of_mass for c in normalized_cycles], axis=0)
        else:
            normalized_angles = np.array([])
            normalized_landmarks = np.array([])
            normalized_com = np.array([])

        return GaitExportPackage(
            metadata=metadata,
            heel_strikes=heel_strikes,
            cycle_info=cycle_info,
            normalized_joint_angles=normalized_angles,
            normalized_landmarks=normalized_landmarks,
            normalized_center_of_mass=normalized_com,
            contralateral_timings=contra_timings,
            angle_names=ANGLE_NAMES.copy(),
            landmark_names=LANDMARK_NAMES.copy(),
        )

    def export_numpy(self, package: GaitExportPackage, path: str):
        """
        Export as NumPy .npz compressed archive.

        Args:
            package: GaitExportPackage to export
            path: Output path (will add .npz extension if not present)
        """
        if not path.endswith('.npz'):
            path = path + '.npz'

        # Convert metadata to dictionary
        metadata_dict = asdict(package.metadata)

        np.savez_compressed(
            path,
            # Main arrays
            joint_angles=package.normalized_joint_angles,
            landmarks=package.normalized_landmarks,
            center_of_mass=package.normalized_center_of_mass,
            contralateral_timings=package.contralateral_timings,

            # Metadata stored as object arrays
            metadata=np.array([metadata_dict], dtype=object),
            heel_strikes=np.array(package.heel_strikes, dtype=object),
            cycle_info=np.array(package.cycle_info, dtype=object),
            angle_names=np.array(package.angle_names),
            landmark_names=np.array(package.landmark_names),
        )

    def export_pickle(self, package: GaitExportPackage, path: str):
        """
        Export as pickle file preserving full object structure.

        Args:
            package: GaitExportPackage to export
            path: Output path (will add .pkl extension if not present)
        """
        if not path.endswith('.pkl'):
            path = path + '.pkl'

        with open(path, 'wb') as f:
            pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)

    def export_csv_angles(self, package: GaitExportPackage, path: str):
        """
        Export joint angles to CSV in tidy format.

        Creates a CSV with columns:
        cycle_id, contralateral_timing, time_percent, angle_name, angle_value

        Args:
            package: GaitExportPackage to export
            path: Output path (will add .csv extension if not present)
        """
        if not path.endswith('.csv'):
            path = path + '.csv'

        rows = []

        for cycle_idx, cycle in enumerate(package.cycle_info):
            cycle_id = cycle['cycle_id']
            contra_timing = cycle.get('contralateral_timing', 50.0)

            for time_idx in range(100):
                for angle_idx, angle_name in enumerate(package.angle_names):
                    angle_value = package.normalized_joint_angles[cycle_idx, time_idx, angle_idx]

                    if not np.isnan(angle_value):
                        rows.append({
                            'cycle_id': cycle_id,
                            'contralateral_timing': contra_timing,
                            'time_percent': time_idx,
                            'angle_name': angle_name,
                            'angle_value': angle_value
                        })

        # Write CSV
        if rows:
            with open(path, 'w') as f:
                # Header
                f.write('cycle_id,contralateral_timing,time_percent,angle_name,angle_value\n')
                # Data
                for row in rows:
                    f.write(f"{row['cycle_id']},{row['contralateral_timing']:.2f},{row['time_percent']},"
                           f"{row['angle_name']},{row['angle_value']:.4f}\n")

    def export_csv_com(self, package: GaitExportPackage, path: str):
        """
        Export center of mass data to CSV.

        Creates a CSV with columns:
        cycle_id, contralateral_timing, time_percent, com_x, com_y

        Args:
            package: GaitExportPackage to export
            path: Output path (will add .csv extension if not present)
        """
        if not path.endswith('.csv'):
            path = path + '.csv'

        rows = []

        for cycle_idx, cycle in enumerate(package.cycle_info):
            cycle_id = cycle['cycle_id']
            contra_timing = cycle.get('contralateral_timing', 50.0)

            for time_idx in range(100):
                com_x = package.normalized_center_of_mass[cycle_idx, time_idx, 0]
                com_y = package.normalized_center_of_mass[cycle_idx, time_idx, 1]

                if not (np.isnan(com_x) or np.isnan(com_y)):
                    rows.append({
                        'cycle_id': cycle_id,
                        'contralateral_timing': contra_timing,
                        'time_percent': time_idx,
                        'com_x': com_x,
                        'com_y': com_y
                    })

        # Write CSV
        if rows:
            with open(path, 'w') as f:
                f.write('cycle_id,contralateral_timing,time_percent,com_x,com_y\n')
                for row in rows:
                    f.write(f"{row['cycle_id']},{row['contralateral_timing']:.2f},{row['time_percent']},"
                           f"{row['com_x']:.6f},{row['com_y']:.6f}\n")

    def export_summary_json(self, package: GaitExportPackage, path: str):
        """
        Export a summary JSON file with metadata and statistics.

        Args:
            package: GaitExportPackage to export
            path: Output path (will add .json extension if not present)
        """
        if not path.endswith('.json'):
            path = path + '.json'

        # Calculate statistics
        stats = {}

        if package.normalized_joint_angles.size > 0:
            for angle_idx, angle_name in enumerate(package.angle_names):
                angle_data = package.normalized_joint_angles[:, :, angle_idx]
                valid_data = angle_data[~np.isnan(angle_data)]
                if len(valid_data) > 0:
                    stats[angle_name] = {
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data))
                    }

        summary = {
            'metadata': asdict(package.metadata),
            'heel_strikes': package.heel_strikes,
            'cycles': package.cycle_info,
            'angle_statistics': stats,
            'angle_names': package.angle_names,
        }

        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

    def export_all(self, package: GaitExportPackage, base_path: str):
        """
        Export data in all available formats.

        Creates:
        - {base_path}.npz - NumPy archive
        - {base_path}.pkl - Pickle file
        - {base_path}_angles.csv - Joint angles CSV
        - {base_path}_com.csv - Center of mass CSV
        - {base_path}_summary.json - Summary JSON

        Args:
            package: GaitExportPackage to export
            base_path: Base path without extension
        """
        self.export_numpy(package, base_path)
        self.export_pickle(package, base_path)
        self.export_csv_angles(package, f"{base_path}_angles")
        self.export_csv_com(package, f"{base_path}_com")
        self.export_summary_json(package, f"{base_path}_summary")

    # ==============================
    # Dual-View Export Methods
    # ==============================

    def create_dual_view_export_package(
        self,
        session: DualViewSession,
        sagittal_results: BatchProcessingResults,
        frontal_results: BatchProcessingResults,
        sagittal_normalized: List[NormalizedGaitData],
        frontal_normalized: List[NormalizedGaitData]
    ) -> DualViewExportPackage:
        """
        Create a complete dual-view export package from a gait analysis session.

        Sagittal view data includes duration, asymmetry, and arm swing metrics.
        Frontal view data includes normalized angles only.

        Args:
            session: DualViewSession with subject metadata
            sagittal_results: BatchProcessingResults from sagittal video
            frontal_results: BatchProcessingResults from frontal video
            sagittal_normalized: List of normalized cycles from sagittal view
            frontal_normalized: List of normalized cycles from frontal view

        Returns:
            DualViewExportPackage ready for export
        """
        # Calculate sagittal statistics
        sagittal_contra_timings = np.array(
            [c.contralateral_timing for c in sagittal_normalized]
        ) if sagittal_normalized else np.array([])

        sagittal_durations = [c.duration_seconds for c in sagittal_normalized]
        sagittal_mean_duration = float(np.mean(sagittal_durations)) if sagittal_durations else 0.0

        sagittal_asymmetries = [abs(c.contralateral_timing - 50.0) for c in sagittal_normalized]
        sagittal_mean_asymmetry = float(np.mean(sagittal_asymmetries)) if sagittal_asymmetries else 0.0

        sagittal_arm_swings = [c.arm_swing_amplitude for c in sagittal_normalized if c.arm_swing_amplitude > 0]
        sagittal_mean_arm_swing = float(np.mean(sagittal_arm_swings)) if sagittal_arm_swings else 0.0

        sagittal_mean_contra = float(np.nanmean(sagittal_contra_timings)) if len(sagittal_contra_timings) > 0 else 50.0

        # Calculate frontal statistics
        frontal_contra_timings = np.array(
            [c.contralateral_timing for c in frontal_normalized]
        ) if frontal_normalized else np.array([])

        # Create metadata
        metadata = DualViewExportMetadata(
            subject_id=session.metadata.subject_id,
            date=session.metadata.date,
            walking_condition=session.metadata.walking_condition,
            export_timestamp=datetime.now().isoformat(),
            sagittal_video_path=sagittal_results.video_path,
            sagittal_video_filename=os.path.basename(sagittal_results.video_path),
            sagittal_camera_side=session.sagittal_camera_side,
            sagittal_fps=sagittal_results.fps,
            sagittal_total_frames=sagittal_results.total_frames,
            sagittal_num_cycles=len(sagittal_normalized),
            sagittal_mean_contralateral_timing=sagittal_mean_contra,
            sagittal_mean_duration=sagittal_mean_duration,
            sagittal_mean_asymmetry=sagittal_mean_asymmetry,
            sagittal_mean_arm_swing=sagittal_mean_arm_swing,
            frontal_video_path=frontal_results.video_path,
            frontal_video_filename=os.path.basename(frontal_results.video_path),
            frontal_fps=frontal_results.fps,
            frontal_total_frames=frontal_results.total_frames,
            frontal_num_cycles=len(frontal_normalized),
        )

        # Serialize sagittal heel strikes
        sagittal_heel_strikes = [
            {
                'frame_idx': e.frame_idx,
                'time_seconds': e.time_seconds,
                'side': e.side,
                'confidence': e.confidence,
                'is_manual': e.is_manual
            }
            for e in sagittal_results.heel_strike_events
        ]

        # Serialize sagittal cycle info (with duration, asymmetry, arm swing)
        sagittal_cycle_info = [
            {
                'cycle_id': c.cycle_id,
                'start_frame': c.start_frame,
                'end_frame': c.end_frame,
                'duration_seconds': c.duration_seconds,
                'contralateral_timing': c.contralateral_timing,
                'contralateral_frame': c.contralateral_frame,
                'step_asymmetry': c.contralateral_timing - 50.0,
                'arm_swing_amplitude': c.arm_swing_amplitude,
                'arm_swing_peak_forward_velocity': c.arm_swing_peak_forward_velocity,
                'arm_swing_peak_backward_velocity': c.arm_swing_peak_backward_velocity,
            }
            for c in sagittal_normalized
        ]

        # Serialize frontal heel strikes
        frontal_heel_strikes = [
            {
                'frame_idx': e.frame_idx,
                'time_seconds': e.time_seconds,
                'side': e.side,
                'confidence': e.confidence,
                'is_manual': e.is_manual
            }
            for e in frontal_results.heel_strike_events
        ]

        # Serialize frontal cycle info (basic info only, no arm swing)
        frontal_cycle_info = [
            {
                'cycle_id': c.cycle_id,
                'start_frame': c.start_frame,
                'end_frame': c.end_frame,
                'duration_seconds': c.duration_seconds,
                'contralateral_timing': c.contralateral_timing,
                'contralateral_frame': c.contralateral_frame,
            }
            for c in frontal_normalized
        ]

        # Stack sagittal normalized arrays
        if sagittal_normalized:
            sagittal_angles = np.stack([c.joint_angles for c in sagittal_normalized], axis=0)
            sagittal_landmarks = np.stack([c.landmarks for c in sagittal_normalized], axis=0)
            sagittal_com = np.stack([c.center_of_mass for c in sagittal_normalized], axis=0)
        else:
            sagittal_angles = np.array([])
            sagittal_landmarks = np.array([])
            sagittal_com = np.array([])

        # Stack frontal normalized arrays
        if frontal_normalized:
            frontal_angles = np.stack([c.joint_angles for c in frontal_normalized], axis=0)
            frontal_landmarks = np.stack([c.landmarks for c in frontal_normalized], axis=0)
            frontal_com = np.stack([c.center_of_mass for c in frontal_normalized], axis=0)
        else:
            frontal_angles = np.array([])
            frontal_landmarks = np.array([])
            frontal_com = np.array([])

        return DualViewExportPackage(
            metadata=metadata,
            sagittal_heel_strikes=sagittal_heel_strikes,
            sagittal_cycle_info=sagittal_cycle_info,
            sagittal_normalized_joint_angles=sagittal_angles,
            sagittal_normalized_landmarks=sagittal_landmarks,
            sagittal_normalized_center_of_mass=sagittal_com,
            sagittal_contralateral_timings=sagittal_contra_timings,
            frontal_heel_strikes=frontal_heel_strikes,
            frontal_cycle_info=frontal_cycle_info,
            frontal_normalized_joint_angles=frontal_angles,
            frontal_normalized_landmarks=frontal_landmarks,
            frontal_normalized_center_of_mass=frontal_com,
            frontal_contralateral_timings=frontal_contra_timings,
            angle_names=ANGLE_NAMES.copy(),
            landmark_names=LANDMARK_NAMES.copy(),
        )

    def export_dual_view_numpy(self, package: DualViewExportPackage, path: str):
        """
        Export dual-view data as NumPy .npz compressed archive.

        Args:
            package: DualViewExportPackage to export
            path: Output path (will add .npz extension if not present)
        """
        if not path.endswith('.npz'):
            path = path + '.npz'

        # Convert metadata to dictionary
        metadata_dict = asdict(package.metadata)

        np.savez_compressed(
            path,
            # Metadata
            metadata=np.array([metadata_dict], dtype=object),

            # Sagittal view arrays
            sagittal_joint_angles=package.sagittal_normalized_joint_angles,
            sagittal_landmarks=package.sagittal_normalized_landmarks,
            sagittal_center_of_mass=package.sagittal_normalized_center_of_mass,
            sagittal_contralateral_timings=package.sagittal_contralateral_timings,
            sagittal_heel_strikes=np.array(package.sagittal_heel_strikes, dtype=object),
            sagittal_cycle_info=np.array(package.sagittal_cycle_info, dtype=object),

            # Frontal view arrays
            frontal_joint_angles=package.frontal_normalized_joint_angles,
            frontal_landmarks=package.frontal_normalized_landmarks,
            frontal_center_of_mass=package.frontal_normalized_center_of_mass,
            frontal_contralateral_timings=package.frontal_contralateral_timings,
            frontal_heel_strikes=np.array(package.frontal_heel_strikes, dtype=object),
            frontal_cycle_info=np.array(package.frontal_cycle_info, dtype=object),

            # Labels
            angle_names=np.array(package.angle_names),
            landmark_names=np.array(package.landmark_names),
        )

    def export_dual_view_pickle(self, package: DualViewExportPackage, path: str):
        """
        Export dual-view data as pickle file preserving full object structure.

        Args:
            package: DualViewExportPackage to export
            path: Output path (will add .pkl extension if not present)
        """
        if not path.endswith('.pkl'):
            path = path + '.pkl'

        with open(path, 'wb') as f:
            pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)

    def export_dual_view_summary_json(self, package: DualViewExportPackage, path: str):
        """
        Export a summary JSON file for dual-view analysis.

        Args:
            package: DualViewExportPackage to export
            path: Output path (will add .json extension if not present)
        """
        if not path.endswith('.json'):
            path = path + '.json'

        # Build summary
        summary = {
            'subject_metadata': {
                'subject_id': package.metadata.subject_id,
                'date': package.metadata.date,
                'walking_condition': package.metadata.walking_condition,
            },
            'export_timestamp': package.metadata.export_timestamp,
            'sagittal_view': {
                'video_filename': package.metadata.sagittal_video_filename,
                'camera_side': package.metadata.sagittal_camera_side,
                'fps': package.metadata.sagittal_fps,
                'total_frames': package.metadata.sagittal_total_frames,
                'num_cycles': package.metadata.sagittal_num_cycles,
                'mean_contralateral_timing': package.metadata.sagittal_mean_contralateral_timing,
                'mean_duration_seconds': package.metadata.sagittal_mean_duration,
                'mean_step_asymmetry_percent': package.metadata.sagittal_mean_asymmetry,
                'mean_arm_swing_degrees': package.metadata.sagittal_mean_arm_swing,
                'cycles': package.sagittal_cycle_info,
            },
            'frontal_view': {
                'video_filename': package.metadata.frontal_video_filename,
                'fps': package.metadata.frontal_fps,
                'total_frames': package.metadata.frontal_total_frames,
                'num_cycles': package.metadata.frontal_num_cycles,
                'cycles': package.frontal_cycle_info,
            },
            'angle_names': package.angle_names,
        }

        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

    def export_dual_view_all(self, package: DualViewExportPackage, base_path: str):
        """
        Export dual-view data in all available formats.

        Creates:
        - {base_path}_dual_view.npz - NumPy archive
        - {base_path}_dual_view.pkl - Pickle file
        - {base_path}_dual_view_summary.json - Summary JSON

        Args:
            package: DualViewExportPackage to export
            base_path: Base path without extension
        """
        self.export_dual_view_numpy(package, f"{base_path}_dual_view")
        self.export_dual_view_pickle(package, f"{base_path}_dual_view")
        self.export_dual_view_summary_json(package, f"{base_path}_dual_view_summary")

    # ========================
    # ML-Ready Export Methods
    # ========================

    def create_ml_dataset(
        self,
        normalized_cycles: List[NormalizedGaitData],
        view_type: str = 'side',
        subject_id: str = None,
        session_id: str = None,
        recording_date: str = None
    ) -> MLGaitDataset:
        """
        Create an ML-ready dataset from normalized gait cycles.

        Args:
            normalized_cycles: List of NormalizedGaitData objects
            view_type: 'side' or 'front'
            subject_id: Optional subject identifier
            session_id: Optional session identifier
            recording_date: Optional recording date string

        Returns:
            MLGaitDataset ready for machine learning
        """
        extractor = GaitFeatureExtractor(view_type=view_type)
        dataset = extractor.extract_features(normalized_cycles)

        # Add optional metadata
        dataset.subject_id = subject_id
        dataset.session_id = session_id
        dataset.recording_date = recording_date or datetime.now().isoformat()

        return dataset

    def export_ml_numpy(self, dataset: MLGaitDataset, path: str):
        """
        Export ML dataset as NumPy .npz archive.

        This is the recommended format for Python/NumPy ML workflows.

        Args:
            dataset: MLGaitDataset to export
            path: Output path (will add .npz extension if not present)
        """
        if not path.endswith('.npz'):
            path = path + '.npz'

        np.savez_compressed(
            path,
            # Main feature matrix
            feature_matrix=dataset.feature_matrix,
            feature_names=np.array(dataset.feature_names, dtype=object),

            # Separate matrices
            time_series_matrix=dataset.time_series_matrix,
            time_series_feature_names=np.array(dataset.time_series_feature_names, dtype=object),
            summary_matrix=dataset.summary_matrix,
            summary_feature_names=np.array(dataset.summary_feature_names, dtype=object),

            # Labels and metadata
            cycle_ids=dataset.cycle_ids,
            contralateral_timings=dataset.contralateral_timings,
            durations=dataset.durations,

            # Dataset info
            n_cycles=dataset.n_cycles,
            n_features=dataset.n_features,
            view_type=dataset.view_type,
            subject_id=dataset.subject_id or '',
            session_id=dataset.session_id or '',
            recording_date=dataset.recording_date or '',

            # Normalization params if computed
            feature_means=dataset.feature_means if dataset.feature_means is not None else np.array([]),
            feature_stds=dataset.feature_stds if dataset.feature_stds is not None else np.array([]),
        )

    def export_ml_csv(self, dataset: MLGaitDataset, path: str, include_header: bool = True):
        """
        Export ML feature matrix as CSV.

        Format suitable for R, SPSS, Excel, or other tools.
        Each row is one gait cycle, each column is a feature.

        Args:
            dataset: MLGaitDataset to export
            path: Output path (will add .csv extension if not present)
            include_header: Whether to include feature names as header row
        """
        if not path.endswith('.csv'):
            path = path + '.csv'

        with open(path, 'w') as f:
            # Write header
            if include_header:
                # Include metadata columns
                header = ['cycle_id', 'contralateral_timing', 'duration_seconds'] + dataset.feature_names
                f.write(','.join(header) + '\n')

            # Write data rows
            for i in range(dataset.n_cycles):
                row_values = [
                    str(dataset.cycle_ids[i]),
                    f"{dataset.contralateral_timings[i]:.2f}",
                    f"{dataset.durations[i]:.4f}"
                ]
                # Add feature values
                for val in dataset.feature_matrix[i]:
                    if np.isnan(val):
                        row_values.append('')
                    else:
                        row_values.append(f"{val:.6f}")

                f.write(','.join(row_values) + '\n')

    def export_ml_parquet(self, dataset: MLGaitDataset, path: str):
        """
        Export ML dataset as Parquet file.

        Efficient columnar format for large-scale data processing.
        Requires pyarrow library.

        Args:
            dataset: MLGaitDataset to export
            path: Output path (will add .parquet extension if not present)
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet export. "
                "Install with: pip install pyarrow"
            )

        if not path.endswith('.parquet'):
            path = path + '.parquet'

        # Build column data
        columns = {
            'cycle_id': dataset.cycle_ids,
            'contralateral_timing': dataset.contralateral_timings,
            'duration_seconds': dataset.durations,
        }

        # Add feature columns
        for i, name in enumerate(dataset.feature_names):
            columns[name] = dataset.feature_matrix[:, i]

        # Create PyArrow table and write
        table = pa.table(columns)
        pq.write_table(table, path, compression='snappy')

    def export_ml_pickle(self, dataset: MLGaitDataset, path: str):
        """
        Export ML dataset as pickle file.

        Preserves the full MLGaitDataset object structure.

        Args:
            dataset: MLGaitDataset to export
            path: Output path (will add .pkl extension if not present)
        """
        if not path.endswith('.pkl'):
            path = path + '.pkl'

        with open(path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    def export_gait_signature(
        self,
        dataset: MLGaitDataset,
        path: str,
        symmetric_only: bool = False,
        asymmetry_threshold: float = 5.0
    ):
        """
        Export a gait signature (baseline statistics) as JSON.

        The signature contains mean, std, median, and IQR for each feature,
        useful for:
        - Establishing individual baseline patterns
        - Detecting deviations in future recordings
        - Comparing between subjects

        Args:
            dataset: MLGaitDataset to compute signature from
            path: Output path (will add .json extension if not present)
            symmetric_only: If True, only include cycles with symmetric timing
            asymmetry_threshold: Threshold for symmetric timing (deviation from 50%)
        """
        if not path.endswith('.json'):
            path = path + '.json'

        signature = compute_gait_signature(
            dataset,
            symmetric_only=symmetric_only,
            asymmetry_threshold=asymmetry_threshold
        )

        # Convert numpy arrays to lists for JSON serialization
        export_data = {
            'subject_id': dataset.subject_id,
            'session_id': dataset.session_id,
            'recording_date': dataset.recording_date,
            'view_type': dataset.view_type,
            'symmetric_only': symmetric_only,
            'asymmetry_threshold': asymmetry_threshold,
            'n_cycles': signature.get('n_cycles', 0),
            'mean_contralateral_timing': signature.get('mean_contralateral_timing'),
            'std_contralateral_timing': signature.get('std_contralateral_timing'),
            'step_asymmetry_stats': signature.get('step_asymmetry_stats', {}),
            'feature_names': signature.get('feature_names', []),
            'statistics': {}
        }

        if 'mean' in signature:
            for i, name in enumerate(signature['feature_names']):
                export_data['statistics'][name] = {
                    'mean': float(signature['mean'][i]) if not np.isnan(signature['mean'][i]) else None,
                    'std': float(signature['std'][i]) if not np.isnan(signature['std'][i]) else None,
                    'median': float(signature['median'][i]) if not np.isnan(signature['median'][i]) else None,
                    'iqr': float(signature['iqr'][i]) if not np.isnan(signature['iqr'][i]) else None,
                }

        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def export_ml_all(
        self,
        normalized_cycles: List[NormalizedGaitData],
        base_path: str,
        view_type: str = 'side',
        subject_id: str = None,
        session_id: str = None
    ):
        """
        Export ML-ready data in all available formats.

        Creates:
        - {base_path}_ml.npz - NumPy archive (recommended for Python)
        - {base_path}_ml.csv - CSV file (for R, SPSS, Excel)
        - {base_path}_ml.pkl - Pickle file (full object)
        - {base_path}_signature.json - Gait signature statistics

        Args:
            normalized_cycles: List of NormalizedGaitData objects
            base_path: Base path without extension
            view_type: 'side' or 'front'
            subject_id: Optional subject identifier
            session_id: Optional session identifier
        """
        # Create dataset
        dataset = self.create_ml_dataset(
            normalized_cycles,
            view_type=view_type,
            subject_id=subject_id,
            session_id=session_id
        )

        # Compute normalization parameters
        dataset.compute_normalization_params()

        # Export in all formats
        self.export_ml_numpy(dataset, f"{base_path}_ml")
        self.export_ml_csv(dataset, f"{base_path}_ml")
        self.export_ml_pickle(dataset, f"{base_path}_ml")
        self.export_gait_signature(dataset, f"{base_path}_signature")

        return dataset


def load_numpy_export(path: str) -> Dict[str, Any]:
    """
    Load a NumPy export file.

    Args:
        path: Path to .npz file

    Returns:
        Dictionary with all exported arrays and metadata
    """
    data = np.load(path, allow_pickle=True)

    return {
        'joint_angles': data['joint_angles'],
        'landmarks': data['landmarks'],
        'center_of_mass': data['center_of_mass'],
        'contralateral_timings': data['contralateral_timings'] if 'contralateral_timings' in data else np.array([]),
        'metadata': data['metadata'][0] if len(data['metadata']) > 0 else {},
        'heel_strikes': list(data['heel_strikes']) if 'heel_strikes' in data else [],
        'cycle_info': list(data['cycle_info']) if 'cycle_info' in data else [],
        'angle_names': list(data['angle_names']) if 'angle_names' in data else [],
        'landmark_names': list(data['landmark_names']) if 'landmark_names' in data else [],
    }


def load_pickle_export(path: str) -> GaitExportPackage:
    """
    Load a pickle export file.

    Args:
        path: Path to .pkl file

    Returns:
        GaitExportPackage object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_ml_numpy_export(path: str) -> MLGaitDataset:
    """
    Load an ML NumPy export file.

    Args:
        path: Path to .npz file

    Returns:
        MLGaitDataset object
    """
    data = np.load(path, allow_pickle=True)

    # Reconstruct MLGaitDataset
    dataset = MLGaitDataset(
        feature_matrix=data['feature_matrix'],
        feature_names=list(data['feature_names']),
        time_series_matrix=data['time_series_matrix'],
        time_series_feature_names=list(data['time_series_feature_names']),
        summary_matrix=data['summary_matrix'],
        summary_feature_names=list(data['summary_feature_names']),
        cycle_ids=data['cycle_ids'],
        contralateral_timings=data['contralateral_timings'],
        durations=data['durations'],
        n_cycles=int(data['n_cycles']),
        n_features=int(data['n_features']),
        view_type=str(data['view_type']),
        subject_id=str(data['subject_id']) if data['subject_id'] else None,
        session_id=str(data['session_id']) if data['session_id'] else None,
        recording_date=str(data['recording_date']) if data['recording_date'] else None,
    )

    # Restore normalization params if present
    if len(data['feature_means']) > 0:
        dataset.feature_means = data['feature_means']
    if len(data['feature_stds']) > 0:
        dataset.feature_stds = data['feature_stds']

    return dataset


def load_ml_pickle_export(path: str) -> MLGaitDataset:
    """
    Load an ML pickle export file.

    Args:
        path: Path to .pkl file

    Returns:
        MLGaitDataset object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_dual_view_numpy_export(path: str) -> Dict[str, Any]:
    """
    Load a dual-view NumPy export file.

    Args:
        path: Path to .npz file

    Returns:
        Dictionary with sagittal and frontal view data
    """
    data = np.load(path, allow_pickle=True)

    return {
        'metadata': data['metadata'][0] if len(data['metadata']) > 0 else {},
        'sagittal': {
            'joint_angles': data['sagittal_joint_angles'],
            'landmarks': data['sagittal_landmarks'],
            'center_of_mass': data['sagittal_center_of_mass'],
            'contralateral_timings': data['sagittal_contralateral_timings'],
            'heel_strikes': list(data['sagittal_heel_strikes']),
            'cycle_info': list(data['sagittal_cycle_info']),
        },
        'frontal': {
            'joint_angles': data['frontal_joint_angles'],
            'landmarks': data['frontal_landmarks'],
            'center_of_mass': data['frontal_center_of_mass'],
            'contralateral_timings': data['frontal_contralateral_timings'],
            'heel_strikes': list(data['frontal_heel_strikes']),
            'cycle_info': list(data['frontal_cycle_info']),
        },
        'angle_names': list(data['angle_names']),
        'landmark_names': list(data['landmark_names']),
    }


def load_dual_view_pickle_export(path: str) -> DualViewExportPackage:
    """
    Load a dual-view pickle export file.

    Args:
        path: Path to .pkl file

    Returns:
        DualViewExportPackage object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
