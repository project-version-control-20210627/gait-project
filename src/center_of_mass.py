"""
Center of Mass calculation module using segmental analysis.

Calculates whole-body center of mass from pose landmarks using
Winter's (2009) body segment model with anthropometric mass fractions.
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np


@dataclass
class SegmentDefinition:
    """Definition of a body segment for CoM calculation."""
    proximal: str | Tuple[str, str]  # Landmark name or tuple of names for midpoint
    distal: str | Tuple[str, str]
    mass_fraction: float
    com_ratio: float  # Distance from proximal as fraction of segment length


class CenterOfMassCalculator:
    """
    Calculates whole-body center of mass using segmental analysis.

    Based on Winter, D.A. (2009) "Biomechanics and Motor Control of Human Movement"
    Uses body segment mass percentages and CoM location ratios from proximal end.
    """

    # Body segment definitions with mass fractions from Winter (2009)
    # com_ratio is the position of segment CoM as fraction from proximal end
    SEGMENT_DEFINITIONS: Dict[str, SegmentDefinition] = {
        'head': SegmentDefinition(
            proximal='nose',  # Approximation - using nose as head reference
            distal='nose',    # Single point for head
            mass_fraction=0.0810,
            com_ratio=0.50
        ),
        'trunk': SegmentDefinition(
            proximal=('left_shoulder', 'right_shoulder'),  # Shoulder midpoint
            distal=('left_hip', 'right_hip'),              # Hip midpoint
            mass_fraction=0.4970,
            com_ratio=0.50
        ),
        'left_upper_arm': SegmentDefinition(
            proximal='left_shoulder',
            distal='left_elbow',
            mass_fraction=0.0280,
            com_ratio=0.436
        ),
        'right_upper_arm': SegmentDefinition(
            proximal='right_shoulder',
            distal='right_elbow',
            mass_fraction=0.0280,
            com_ratio=0.436
        ),
        'left_forearm': SegmentDefinition(
            proximal='left_elbow',
            distal='left_wrist',
            mass_fraction=0.0160,
            com_ratio=0.430
        ),
        'right_forearm': SegmentDefinition(
            proximal='right_elbow',
            distal='right_wrist',
            mass_fraction=0.0160,
            com_ratio=0.430
        ),
        'left_hand': SegmentDefinition(
            proximal='left_wrist',
            distal='left_index',  # Approximation
            mass_fraction=0.0060,
            com_ratio=0.506
        ),
        'right_hand': SegmentDefinition(
            proximal='right_wrist',
            distal='right_index',
            mass_fraction=0.0060,
            com_ratio=0.506
        ),
        'left_thigh': SegmentDefinition(
            proximal='left_hip',
            distal='left_knee',
            mass_fraction=0.1000,
            com_ratio=0.433
        ),
        'right_thigh': SegmentDefinition(
            proximal='right_hip',
            distal='right_knee',
            mass_fraction=0.1000,
            com_ratio=0.433
        ),
        'left_shank': SegmentDefinition(
            proximal='left_knee',
            distal='left_ankle',
            mass_fraction=0.0465,
            com_ratio=0.433
        ),
        'right_shank': SegmentDefinition(
            proximal='right_knee',
            distal='right_ankle',
            mass_fraction=0.0465,
            com_ratio=0.433
        ),
        'left_foot': SegmentDefinition(
            proximal='left_heel',
            distal='left_foot_index',
            mass_fraction=0.0145,
            com_ratio=0.50
        ),
        'right_foot': SegmentDefinition(
            proximal='right_heel',
            distal='right_foot_index',
            mass_fraction=0.0145,
            com_ratio=0.50
        ),
    }

    # Minimum visibility threshold for landmarks
    MIN_VISIBILITY = 0.3

    # Minimum mass fraction required for valid CoM calculation
    MIN_MASS_FRACTION = 0.5

    def __init__(self, min_visibility: float = 0.3, min_mass_fraction: float = 0.5):
        """
        Initialize the calculator.

        Args:
            min_visibility: Minimum landmark visibility threshold (0-1)
            min_mass_fraction: Minimum fraction of body mass needed for valid CoM
        """
        self.min_visibility = min_visibility
        self.min_mass_fraction = min_mass_fraction

    def calculate_com(
        self,
        landmarks: Dict[str, Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate whole-body center of mass from landmarks.

        Args:
            landmarks: Dictionary mapping landmark names to (x, y, visibility) tuples
                      where x and y are normalized coordinates (0-1).

        Returns:
            (x, y) tuple in normalized coordinates, or None if insufficient landmarks.
        """
        total_weighted_x = 0.0
        total_weighted_y = 0.0
        total_mass_fraction = 0.0

        for segment_name, segment_def in self.SEGMENT_DEFINITIONS.items():
            segment_com = self._calculate_segment_com(landmarks, segment_def)

            if segment_com is not None:
                total_weighted_x += segment_com[0] * segment_def.mass_fraction
                total_weighted_y += segment_com[1] * segment_def.mass_fraction
                total_mass_fraction += segment_def.mass_fraction

        if total_mass_fraction < self.min_mass_fraction:
            return None

        # Normalize by actual mass fraction used
        return (
            total_weighted_x / total_mass_fraction,
            total_weighted_y / total_mass_fraction
        )

    def calculate_segment_coms(
        self,
        landmarks: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Calculate center of mass for each body segment.

        Args:
            landmarks: Dictionary of normalized landmarks

        Returns:
            Dictionary mapping segment names to (x, y) CoM positions or None
        """
        segment_coms = {}
        for segment_name, segment_def in self.SEGMENT_DEFINITIONS.items():
            segment_coms[segment_name] = self._calculate_segment_com(landmarks, segment_def)
        return segment_coms

    def _calculate_segment_com(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
        segment_def: SegmentDefinition
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate CoM for a single body segment.

        Args:
            landmarks: Dictionary of normalized landmarks
            segment_def: Segment definition with endpoints and CoM ratio

        Returns:
            (x, y) CoM position or None if landmarks not visible
        """
        # Get proximal endpoint
        prox_pos = self._get_endpoint_position(landmarks, segment_def.proximal)
        if prox_pos is None:
            return None

        # Get distal endpoint
        dist_pos = self._get_endpoint_position(landmarks, segment_def.distal)
        if dist_pos is None:
            return None

        # Calculate CoM position along segment
        com_x = prox_pos[0] + segment_def.com_ratio * (dist_pos[0] - prox_pos[0])
        com_y = prox_pos[1] + segment_def.com_ratio * (dist_pos[1] - prox_pos[1])

        return (com_x, com_y)

    def _get_endpoint_position(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
        endpoint: str | Tuple[str, str]
    ) -> Optional[Tuple[float, float]]:
        """
        Get position of a segment endpoint (single landmark or midpoint of two).

        Args:
            landmarks: Dictionary of normalized landmarks
            endpoint: Landmark name or tuple of two landmark names for midpoint

        Returns:
            (x, y) position or None if landmarks not visible
        """
        if isinstance(endpoint, tuple):
            # Midpoint of two landmarks
            lm1 = landmarks.get(endpoint[0])
            lm2 = landmarks.get(endpoint[1])

            if lm1 is None or lm2 is None:
                return None
            if lm1[2] < self.min_visibility or lm2[2] < self.min_visibility:
                return None

            return ((lm1[0] + lm2[0]) / 2, (lm1[1] + lm2[1]) / 2)
        else:
            # Single landmark
            lm = landmarks.get(endpoint)
            if lm is None or lm[2] < self.min_visibility:
                return None
            return (lm[0], lm[1])

    def get_mass_fractions(self) -> Dict[str, float]:
        """Get dictionary of segment mass fractions."""
        return {name: seg.mass_fraction for name, seg in self.SEGMENT_DEFINITIONS.items()}

    def get_total_mass_fraction(self) -> float:
        """Get total mass fraction (should sum to ~1.0)."""
        return sum(seg.mass_fraction for seg in self.SEGMENT_DEFINITIONS.values())


def calculate_com_trajectory(
    landmarks_sequence: List[Dict[str, Tuple[float, float, float]]],
    calculator: CenterOfMassCalculator = None
) -> np.ndarray:
    """
    Calculate center of mass trajectory over a sequence of frames.

    Args:
        landmarks_sequence: List of landmark dictionaries, one per frame
        calculator: Optional pre-configured calculator instance

    Returns:
        Array of shape (n_frames, 2) with CoM (x, y) positions.
        NaN values indicate frames where CoM could not be calculated.
    """
    if calculator is None:
        calculator = CenterOfMassCalculator()

    n_frames = len(landmarks_sequence)
    trajectory = np.full((n_frames, 2), np.nan)

    for i, landmarks in enumerate(landmarks_sequence):
        if landmarks is not None:
            com = calculator.calculate_com(landmarks)
            if com is not None:
                trajectory[i, 0] = com[0]
                trajectory[i, 1] = com[1]

    return trajectory
