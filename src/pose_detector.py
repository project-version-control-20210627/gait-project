"""
Pose detection module using MediaPipe Tasks API.

Provides pose landmark detection and stick figure visualization
for human gait analysis.
"""

from typing import Optional, Tuple, Dict
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseDetector:
    """Detects human pose landmarks using MediaPipe Pose Landmarker."""

    # MediaPipe Pose landmark indices (33 landmarks)
    LANDMARKS = {
        'nose': 0,
        'left_eye_inner': 1,
        'left_eye': 2,
        'left_eye_outer': 3,
        'right_eye_inner': 4,
        'right_eye': 5,
        'right_eye_outer': 6,
        'left_ear': 7,
        'right_ear': 8,
        'mouth_left': 9,
        'mouth_right': 10,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_pinky': 17,
        'right_pinky': 18,
        'left_index': 19,
        'right_index': 20,
        'left_thumb': 21,
        'right_thumb': 22,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_heel': 29,
        'right_heel': 30,
        'left_foot_index': 31,
        'right_foot_index': 32,
    }

    # Connections for stick figure drawing
    # Format: (start_landmark, end_landmark, color_bgr)
    STICK_FIGURE_CONNECTIONS = [
        # Head and neck
        ('nose', 'left_eye', (255, 255, 255)),  # White - face
        ('nose', 'right_eye', (255, 255, 255)),
        ('left_eye', 'left_ear', (255, 255, 255)),
        ('right_eye', 'right_ear', (255, 255, 255)),

        # Spine/torso - using midpoints
        ('left_shoulder', 'right_shoulder', (0, 255, 255)),  # Yellow - shoulders
        ('left_hip', 'right_hip', (0, 255, 255)),  # Yellow - pelvis

        # Left arm (blue tones)
        ('left_shoulder', 'left_elbow', (255, 128, 0)),  # Blue - upper arm
        ('left_elbow', 'left_wrist', (255, 64, 0)),  # Darker blue - forearm

        # Right arm (green tones)
        ('right_shoulder', 'right_elbow', (0, 255, 0)),  # Green - upper arm
        ('right_elbow', 'right_wrist', (0, 200, 0)),  # Darker green - forearm

        # Left leg (red tones)
        ('left_hip', 'left_knee', (0, 0, 255)),  # Red - thigh
        ('left_knee', 'left_ankle', (0, 0, 200)),  # Darker red - lower leg
        ('left_ankle', 'left_heel', (0, 0, 150)),  # Even darker - foot
        ('left_heel', 'left_foot_index', (0, 0, 150)),

        # Right leg (magenta tones)
        ('right_hip', 'right_knee', (255, 0, 255)),  # Magenta - thigh
        ('right_knee', 'right_ankle', (200, 0, 200)),  # Darker magenta - lower leg
        ('right_ankle', 'right_heel', (150, 0, 150)),  # Even darker - foot
        ('right_heel', 'right_foot_index', (150, 0, 150)),
    ]

    def __init__(
        self,
        model_path: str = None,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the pose detector.

        Args:
            model_path: Path to the pose landmarker model file (.task).
                       If None, looks in default locations.
            num_poses: Maximum number of poses to detect.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
        """
        # Find model file
        if model_path is None:
            # Look in common locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'data', 'pose_landmarker_heavy.task'),
                os.path.join(os.path.dirname(__file__), 'pose_landmarker_heavy.task'),
                'pose_landmarker_heavy.task',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path is None:
                raise FileNotFoundError(
                    "Could not find pose_landmarker_heavy.task model file. "
                    "Please download it from: "
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
                )

        # Create pose landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Detect pose landmarks in a frame.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            Dictionary mapping landmark names to (x, y, visibility) tuples,
            where x and y are normalized coordinates (0-1).
            Returns None if no pose detected.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect poses
        results = self.detector.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        # Get first pose
        pose_landmarks = results.pose_landmarks[0]

        landmarks = {}
        for name, idx in self.LANDMARKS.items():
            lm = pose_landmarks[idx]
            landmarks[name] = (lm.x, lm.y, lm.visibility)

        return landmarks

    def get_pixel_coordinates(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
        frame_width: int,
        frame_height: int
    ) -> Dict[str, Tuple[int, int, float]]:
        """
        Convert normalized landmarks to pixel coordinates.

        Args:
            landmarks: Dictionary of normalized landmarks
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels

        Returns:
            Dictionary mapping landmark names to (x, y, visibility) in pixels
        """
        pixel_landmarks = {}
        for name, (x, y, vis) in landmarks.items():
            px = int(x * frame_width)
            py = int(y * frame_height)
            pixel_landmarks[name] = (px, py, vis)
        return pixel_landmarks

    def draw_stick_figure(
        self,
        frame: np.ndarray,
        landmarks: Dict[str, Tuple[float, float, float]],
        draw_on_original: bool = True,
        line_thickness: int = 3,
        circle_radius: int = 5,
        min_visibility: float = 0.5
    ) -> np.ndarray:
        """
        Draw a simplified stick figure on the frame.

        Args:
            frame: BGR image to draw on
            landmarks: Dictionary of normalized landmarks
            draw_on_original: If True, draw on original frame. If False, draw on black background.
            line_thickness: Thickness of stick figure lines
            circle_radius: Radius of joint circles
            min_visibility: Minimum visibility threshold for drawing

        Returns:
            Frame with stick figure drawn
        """
        h, w = frame.shape[:2]

        if draw_on_original:
            output = frame.copy()
        else:
            output = np.zeros_like(frame)

        pixel_landmarks = self.get_pixel_coordinates(landmarks, w, h)

        # Draw torso connections (shoulder midpoint to hip midpoint for spine)
        left_shoulder = pixel_landmarks.get('left_shoulder')
        right_shoulder = pixel_landmarks.get('right_shoulder')
        left_hip = pixel_landmarks.get('left_hip')
        right_hip = pixel_landmarks.get('right_hip')
        nose = pixel_landmarks.get('nose')

        # Calculate midpoints for spine
        if all(v is not None for v in [left_shoulder, right_shoulder, left_hip, right_hip]):
            if all(v[2] >= min_visibility for v in [left_shoulder, right_shoulder]):
                shoulder_mid = (
                    (left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2
                )
            else:
                shoulder_mid = None

            if all(v[2] >= min_visibility for v in [left_hip, right_hip]):
                hip_mid = (
                    (left_hip[0] + right_hip[0]) // 2,
                    (left_hip[1] + right_hip[1]) // 2
                )
            else:
                hip_mid = None

            # Draw spine (torso)
            if shoulder_mid and hip_mid:
                cv2.line(output, shoulder_mid, hip_mid, (0, 255, 255), line_thickness)

            # Draw neck (nose to shoulder midpoint)
            if shoulder_mid and nose and nose[2] >= min_visibility:
                cv2.line(output, (nose[0], nose[1]), shoulder_mid, (255, 255, 255), line_thickness)

            # Draw shoulder to hip connections on each side
            if left_shoulder[2] >= min_visibility and left_hip[2] >= min_visibility:
                cv2.line(output, (left_shoulder[0], left_shoulder[1]),
                        (left_hip[0], left_hip[1]), (128, 128, 128), line_thickness // 2)
            if right_shoulder[2] >= min_visibility and right_hip[2] >= min_visibility:
                cv2.line(output, (right_shoulder[0], right_shoulder[1]),
                        (right_hip[0], right_hip[1]), (128, 128, 128), line_thickness // 2)

        # Draw all standard connections
        for start_name, end_name, color in self.STICK_FIGURE_CONNECTIONS:
            start = pixel_landmarks.get(start_name)
            end = pixel_landmarks.get(end_name)

            if start is None or end is None:
                continue

            # Check visibility
            if start[2] < min_visibility or end[2] < min_visibility:
                continue

            cv2.line(output, (start[0], start[1]), (end[0], end[1]), color, line_thickness)

        # Draw joint circles
        for name, (x, y, vis) in pixel_landmarks.items():
            if vis < min_visibility:
                continue

            # Color based on body part
            if 'shoulder' in name or 'hip' in name:
                color = (0, 255, 255)  # Yellow for major joints
            elif 'elbow' in name or 'knee' in name:
                color = (255, 255, 0)  # Cyan for mid joints
            elif 'wrist' in name or 'ankle' in name:
                color = (255, 0, 255)  # Magenta for distal joints
            else:
                color = (255, 255, 255)  # White for others

            cv2.circle(output, (x, y), circle_radius, color, -1)
            cv2.circle(output, (x, y), circle_radius, (0, 0, 0), 1)  # Black outline

        return output

    def calculate_angle(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
        point1_name: str,
        vertex_name: str,
        point2_name: str
    ) -> Optional[float]:
        """
        Calculate the angle at a vertex formed by three landmarks.

        Args:
            landmarks: Dictionary of normalized landmarks
            point1_name: First point name
            vertex_name: Vertex point name (angle is measured here)
            point2_name: Second point name

        Returns:
            Angle in degrees, or None if landmarks not visible
        """
        p1 = landmarks.get(point1_name)
        vertex = landmarks.get(vertex_name)
        p2 = landmarks.get(point2_name)

        if any(v is None for v in [p1, vertex, p2]):
            return None

        # Check visibility
        if any(v[2] < 0.5 for v in [p1, vertex, p2]):
            return None

        # Calculate vectors
        v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]])
        v2 = np.array([p2[0] - vertex[0], p2[1] - vertex[1]])

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def _calculate_ankle_angle(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
        side: str
    ) -> Optional[float]:
        """
        Calculate ankle dorsiflexion/plantarflexion angle using heel-based foot orientation.

        This method uses the heel landmark to define the foot plane, which is more stable
        than using the foot_index (toe) landmark. The angle represents:
        - 90° = foot perpendicular to tibia (neutral)
        - < 90° = plantarflexion (pointing toes down)
        - > 90° = dorsiflexion (toes pointing up)

        Args:
            landmarks: Dictionary of normalized landmarks
            side: 'left' or 'right'

        Returns:
            Ankle angle in degrees, or None if landmarks not visible
        """
        knee = landmarks.get(f'{side}_knee')
        ankle = landmarks.get(f'{side}_ankle')
        heel = landmarks.get(f'{side}_heel')
        foot_index = landmarks.get(f'{side}_foot_index')

        if any(v is None for v in [knee, ankle, heel, foot_index]):
            return None

        # Check visibility - use lower threshold for foot landmarks as they're often less visible
        if knee[2] < 0.5 or ankle[2] < 0.5:
            return None
        if heel[2] < 0.3 and foot_index[2] < 0.3:
            return None

        # Vector from ankle to knee (tibia direction, pointing up)
        tibia = np.array([knee[0] - ankle[0], knee[1] - ankle[1]])

        # Determine foot vector based on landmark visibility
        # Prefer using heel if visible, fall back to foot_index
        if heel[2] >= 0.3:
            # Vector from heel to ankle, then extend to approximate foot direction
            # The foot direction is from heel towards toes
            if foot_index[2] >= 0.3:
                # Best case: use actual heel-to-toe direction
                foot = np.array([foot_index[0] - heel[0], foot_index[1] - heel[1]])
            else:
                # Use heel-to-ankle as approximation (reverse direction represents foot)
                foot = np.array([ankle[0] - heel[0], ankle[1] - heel[1]])
        else:
            # Fall back to ankle-to-toe direction
            foot = np.array([foot_index[0] - ankle[0], foot_index[1] - ankle[1]])

        # Calculate angle between tibia and foot vectors
        tibia_norm = np.linalg.norm(tibia)
        foot_norm = np.linalg.norm(foot)

        if tibia_norm < 1e-6 or foot_norm < 1e-6:
            return None

        cos_angle = np.dot(tibia, foot) / (tibia_norm * foot_norm)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def get_joint_angles(
        self,
        landmarks: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, Optional[float]]:
        """
        Calculate key joint angles for gait analysis.

        Returns angles for:
        - Neck flexion (head relative to torso)
        - Trunk flexion (torso relative to pelvis)
        - Shoulder angles (both sides)
        - Elbow angles (both sides)
        - Hip angles (both sides)
        - Knee angles (both sides)
        - Ankle angles (both sides)
        """
        angles = {}

        # Left arm angles
        angles['left_elbow'] = self.calculate_angle(
            landmarks, 'left_shoulder', 'left_elbow', 'left_wrist'
        )
        angles['left_shoulder'] = self.calculate_angle(
            landmarks, 'left_elbow', 'left_shoulder', 'left_hip'
        )

        # Right arm angles
        angles['right_elbow'] = self.calculate_angle(
            landmarks, 'right_shoulder', 'right_elbow', 'right_wrist'
        )
        angles['right_shoulder'] = self.calculate_angle(
            landmarks, 'right_elbow', 'right_shoulder', 'right_hip'
        )

        # Left leg angles
        angles['left_hip'] = self.calculate_angle(
            landmarks, 'left_shoulder', 'left_hip', 'left_knee'
        )
        angles['left_knee'] = self.calculate_angle(
            landmarks, 'left_hip', 'left_knee', 'left_ankle'
        )
        angles['left_ankle'] = self._calculate_ankle_angle(landmarks, 'left')

        # Right leg angles
        angles['right_hip'] = self.calculate_angle(
            landmarks, 'right_shoulder', 'right_hip', 'right_knee'
        )
        angles['right_knee'] = self.calculate_angle(
            landmarks, 'right_hip', 'right_knee', 'right_ankle'
        )
        angles['right_ankle'] = self._calculate_ankle_angle(landmarks, 'right')

        return angles

    def get_postural_angles(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
        view_type: str = 'side'
    ) -> Dict[str, Optional[float]]:
        """
        Calculate postural/axial skeleton angles for gait analysis.

        For side view (sagittal plane):
        - cervical_flexion: Forward head posture angle (nose to shoulder midpoint)
          relative to vertical (0° = vertical, positive = forward lean)
        - thoracic_inclination: Upper back posture angle (shoulder midpoint to ear
          midpoint) relative to vertical. Captures upper thoracic kyphosis and
          head-forward posture (0° = vertical, positive = forward lean)
        - trunk_inclination: Overall trunk lean (shoulder midpoint to hip midpoint)
          relative to vertical (0° = vertical, positive = forward lean)

        For frontal view:
        - shoulder_tilt: Angle of shoulder line from horizontal
          (0° = level, positive = right shoulder higher)
        - hip_tilt: Angle of hip/pelvis line from horizontal
          (0° = level, positive = right hip higher)
        - trunk_lateral_lean: Lateral deviation of trunk from vertical
          (0° = vertical, positive = leaning right)

        Args:
            landmarks: Dictionary of normalized landmarks
            view_type: 'side' for sagittal plane or 'front' for frontal plane

        Returns:
            Dictionary of postural angle names to values in degrees
        """
        # Always compute both sagittal and frontal postural angles so that
        # normalized gait cycle data contains all angles regardless of which
        # view_type was selected during batch processing.
        angles = {}
        angles.update(self._calculate_sagittal_postural_angles(landmarks))
        angles.update(self._calculate_frontal_postural_angles(landmarks))

        return angles

    def _calculate_sagittal_postural_angles(
        self,
        landmarks: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, Optional[float]]:
        """
        Calculate postural angles for side (sagittal) view.

        In image coordinates: x increases rightward, y increases downward.
        Vertical reference is [0, 1] (pointing down).

        Angles are measured as deviation from vertical:
        - 0° = perfectly vertical
        - Positive = forward lean (in direction of walking)
        - Negative = backward lean
        """
        angles = {
            'cervical_flexion': None,
            'thoracic_inclination': None,
            'trunk_inclination': None,
        }

        # Get key landmarks
        nose = landmarks.get('nose')
        left_shoulder = landmarks.get('left_shoulder')
        right_shoulder = landmarks.get('right_shoulder')
        left_hip = landmarks.get('left_hip')
        right_hip = landmarks.get('right_hip')
        left_ear = landmarks.get('left_ear')
        right_ear = landmarks.get('right_ear')

        # Calculate midpoints if both sides are visible
        shoulder_mid = None
        hip_mid = None
        ear_mid = None

        if (left_shoulder and right_shoulder and
            left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3):
            shoulder_mid = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            )

        if (left_hip and right_hip and
            left_hip[2] > 0.3 and right_hip[2] > 0.3):
            hip_mid = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )

        if (left_ear and right_ear and
            left_ear[2] > 0.3 and right_ear[2] > 0.3):
            ear_mid = (
                (left_ear[0] + right_ear[0]) / 2,
                (left_ear[1] + right_ear[1]) / 2
            )

        # Cervical flexion: angle of neck (nose to shoulder_mid) from vertical
        if nose and shoulder_mid and nose[2] > 0.5:
            angles['cervical_flexion'] = self._angle_from_vertical(
                shoulder_mid, (nose[0], nose[1])
            )

        # Trunk inclination: angle of full trunk (shoulder_mid to hip_mid) from vertical
        # Represents overall forward/backward lean of the torso
        if shoulder_mid and hip_mid:
            angles['trunk_inclination'] = self._angle_from_vertical(
                hip_mid, shoulder_mid
            )

        # Thoracic inclination: angle of upper back (shoulder_mid to ear_mid) from vertical
        # Captures upper back rounding/posture distinct from overall trunk lean
        # More sensitive to head-forward posture and upper thoracic kyphosis
        if shoulder_mid and ear_mid:
            angles['thoracic_inclination'] = self._angle_from_vertical(
                shoulder_mid, ear_mid
            )

        return angles

    def _calculate_frontal_postural_angles(
        self,
        landmarks: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, Optional[float]]:
        """
        Calculate postural angles for front (frontal) view.

        In image coordinates: x increases rightward, y increases downward.
        Horizontal reference is [1, 0] (pointing right).

        Angles represent tilt from horizontal:
        - 0° = perfectly level
        - Positive = right side higher (for shoulder/hip tilt)
        - Negative = left side higher
        """
        angles = {
            'shoulder_tilt': None,
            'hip_tilt': None,
            'trunk_lateral_lean': None,
        }

        # Get key landmarks
        left_shoulder = landmarks.get('left_shoulder')
        right_shoulder = landmarks.get('right_shoulder')
        left_hip = landmarks.get('left_hip')
        right_hip = landmarks.get('right_hip')

        # Shoulder tilt: angle of shoulder line from horizontal
        if (left_shoulder and right_shoulder and
            left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3):
            # Vector from left to right shoulder
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            # Angle from horizontal (positive if right shoulder is lower in image = higher physically)
            # Note: in image coords, y increases downward, so negative dy means right is higher
            angles['shoulder_tilt'] = -np.degrees(np.arctan2(dy, dx))

        # Hip tilt: angle of hip line from horizontal
        if (left_hip and right_hip and
            left_hip[2] > 0.3 and right_hip[2] > 0.3):
            dx = right_hip[0] - left_hip[0]
            dy = right_hip[1] - left_hip[1]
            angles['hip_tilt'] = -np.degrees(np.arctan2(dy, dx))

        # Trunk lateral lean: deviation of spine from vertical
        # Measured as the angle of the line from hip midpoint to shoulder midpoint
        if (left_shoulder and right_shoulder and left_hip and right_hip and
            left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3 and
            left_hip[2] > 0.3 and right_hip[2] > 0.3):

            shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_mid_x = (left_hip[0] + right_hip[0]) / 2
            hip_mid_y = (left_hip[1] + right_hip[1]) / 2

            # Vector from hip midpoint to shoulder midpoint
            dx = shoulder_mid_x - hip_mid_x
            dy = shoulder_mid_y - hip_mid_y

            # Angle from vertical (positive = leaning right)
            # Vertical in image is [0, -1] (pointing up)
            # Use atan2 to get angle, then adjust for vertical reference
            angle_from_vertical = np.degrees(np.arctan2(dx, -dy))
            angles['trunk_lateral_lean'] = angle_from_vertical

        return angles

    def _angle_from_vertical(
        self,
        base_point: Tuple[float, float],
        top_point: Tuple[float, float]
    ) -> float:
        """
        Calculate the angle of a line segment from vertical.

        Args:
            base_point: (x, y) of the lower point (e.g., hip or shoulder)
            top_point: (x, y) of the upper point (e.g., shoulder or nose)

        Returns:
            Angle in degrees from vertical.
            Positive = forward lean (top point is ahead of base in x)
            Negative = backward lean

        Note: In image coordinates, y increases downward, so "up" is negative y.
        """
        dx = top_point[0] - base_point[0]
        dy = top_point[1] - base_point[1]

        # Vertical reference pointing up is (0, -1) in image coordinates
        # The angle of the vector (dx, dy) from vertical (0, -1)
        # Using atan2: angle from vertical = atan2(dx, -dy)
        angle = np.degrees(np.arctan2(dx, -dy))

        return angle

    def get_all_angles(
        self,
        landmarks: Dict[str, Tuple[float, float, float]],
        view_type: str = 'side'
    ) -> Dict[str, Optional[float]]:
        """
        Get both joint angles and postural angles combined.

        Args:
            landmarks: Dictionary of normalized landmarks
            view_type: 'side' or 'front'

        Returns:
            Combined dictionary of all angle measurements
        """
        angles = self.get_joint_angles(landmarks)
        angles.update(self.get_postural_angles(landmarks, view_type))
        return angles

    def close(self):
        """Release resources."""
        self.detector.close()
