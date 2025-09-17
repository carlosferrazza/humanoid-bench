import enum
from .robot import Robot


class H1(Robot):
    class JOINT(enum.IntEnum):
        left_hip_yaw = 0
        left_hip_roll = 1
        left_hip_pitch = 2
        left_knee = 3
        left_ankle = 4
        right_hip_yaw = 5
        right_hip_roll = 6
        right_hip_pitch = 7
        right_knee = 8
        right_ankle = 9
        torso = 10
        left_shoulder_pitch = 11
        left_shoulder_roll = 12
        left_shoulder_yaw = 13
        left_elbow = 14
        right_shoulder_pitch = 15
        right_shoulder_roll = 16
        right_shoulder_yaw = 17
        right_elbow = 18

    @property
    def joint_connections(self):
        return [
            # left hip yaw
            (self.JOINT.left_hip_yaw, self.JOINT.torso),
            (self.JOINT.left_hip_yaw, self.JOINT.left_hip_roll),
            (self.JOINT.left_hip_yaw, self.JOINT.left_hip_pitch),
            (self.JOINT.left_hip_yaw, self.JOINT.left_knee),
            # left hip roll
            (self.JOINT.left_hip_roll, self.JOINT.torso),
            (self.JOINT.left_hip_roll, self.JOINT.left_hip_yaw),
            (self.JOINT.left_hip_roll, self.JOINT.left_hip_pitch),
            (self.JOINT.left_hip_roll, self.JOINT.left_knee),
            # left hip pitch
            (self.JOINT.left_hip_pitch, self.JOINT.torso),
            (self.JOINT.left_hip_pitch, self.JOINT.left_hip_yaw),
            (self.JOINT.left_hip_pitch, self.JOINT.left_hip_roll),
            (self.JOINT.left_hip_pitch, self.JOINT.left_knee),
            # left knee
            (self.JOINT.left_knee, self.JOINT.left_hip_yaw),
            (self.JOINT.left_knee, self.JOINT.left_hip_roll),
            (self.JOINT.left_knee, self.JOINT.left_hip_pitch),
            (self.JOINT.left_knee, self.JOINT.left_ankle),
            # left ankle
            (self.JOINT.left_ankle, self.JOINT.left_knee),
            # right hip yaw
            (self.JOINT.right_hip_yaw, self.JOINT.torso),
            (self.JOINT.right_hip_yaw, self.JOINT.right_hip_roll),
            (self.JOINT.right_hip_yaw, self.JOINT.right_hip_pitch),
            (self.JOINT.right_hip_yaw, self.JOINT.right_knee),
            # right hip roll
            (self.JOINT.right_hip_roll, self.JOINT.torso),
            (self.JOINT.right_hip_roll, self.JOINT.right_hip_yaw),
            (self.JOINT.right_hip_roll, self.JOINT.right_hip_pitch),
            (self.JOINT.right_hip_roll, self.JOINT.right_knee),
            # right hip pitch
            (self.JOINT.right_hip_pitch, self.JOINT.torso),
            (self.JOINT.right_hip_pitch, self.JOINT.right_hip_yaw),
            (self.JOINT.right_hip_pitch, self.JOINT.right_hip_roll),
            (self.JOINT.right_hip_pitch, self.JOINT.right_knee),
            # right knee
            (self.JOINT.right_knee, self.JOINT.right_hip_yaw),
            (self.JOINT.right_knee, self.JOINT.right_hip_roll),
            (self.JOINT.right_knee, self.JOINT.right_hip_pitch),
            (self.JOINT.right_knee, self.JOINT.right_ankle),
            # right ankle
            (self.JOINT.right_ankle, self.JOINT.right_knee),
            # torso
            (self.JOINT.torso, self.JOINT.left_hip_yaw),
            (self.JOINT.torso, self.JOINT.right_hip_yaw),
            (self.JOINT.torso, self.JOINT.left_hip_roll),
            (self.JOINT.torso, self.JOINT.right_hip_roll),
            (self.JOINT.torso, self.JOINT.left_hip_pitch),
            (self.JOINT.torso, self.JOINT.right_hip_pitch),
            (self.JOINT.torso, self.JOINT.left_shoulder_pitch),
            (self.JOINT.torso, self.JOINT.right_shoulder_pitch),
            (self.JOINT.torso, self.JOINT.left_shoulder_roll),
            (self.JOINT.torso, self.JOINT.right_shoulder_roll),
            (self.JOINT.torso, self.JOINT.left_shoulder_yaw),
            (self.JOINT.torso, self.JOINT.right_shoulder_yaw),
            # left shoulder pitch
            (self.JOINT.left_shoulder_pitch, self.JOINT.torso),
            (self.JOINT.left_shoulder_pitch, self.JOINT.left_shoulder_roll),
            (self.JOINT.left_shoulder_pitch, self.JOINT.left_shoulder_yaw),
            (self.JOINT.left_shoulder_pitch, self.JOINT.left_elbow),
            # left shoulder roll
            (self.JOINT.left_shoulder_roll, self.JOINT.torso),
            (self.JOINT.left_shoulder_roll, self.JOINT.left_shoulder_pitch),
            (self.JOINT.left_shoulder_roll, self.JOINT.left_shoulder_yaw),
            (self.JOINT.left_shoulder_roll, self.JOINT.left_elbow),
            # left shoulder yaw
            (self.JOINT.left_shoulder_yaw, self.JOINT.torso),
            (self.JOINT.left_shoulder_yaw, self.JOINT.left_shoulder_roll),
            (self.JOINT.left_shoulder_yaw, self.JOINT.left_shoulder_pitch),
            (self.JOINT.left_shoulder_yaw, self.JOINT.left_elbow),
            # left elbow
            (self.JOINT.left_elbow, self.JOINT.left_shoulder_roll),
            (self.JOINT.left_elbow, self.JOINT.left_shoulder_pitch),
            (self.JOINT.left_elbow, self.JOINT.left_shoulder_yaw),
            # right shoulder pitch
            (self.JOINT.right_shoulder_pitch, self.JOINT.torso),
            (self.JOINT.right_shoulder_pitch, self.JOINT.right_shoulder_roll),
            (self.JOINT.right_shoulder_pitch, self.JOINT.right_shoulder_yaw),
            (self.JOINT.right_shoulder_pitch, self.JOINT.right_elbow),
            # right shoulder roll
            (self.JOINT.right_shoulder_roll, self.JOINT.torso),
            (self.JOINT.right_shoulder_roll, self.JOINT.right_shoulder_pitch),
            (self.JOINT.right_shoulder_roll, self.JOINT.right_shoulder_yaw),
            (self.JOINT.right_shoulder_roll, self.JOINT.right_elbow),
            # right shoulder yaw
            (self.JOINT.right_shoulder_yaw, self.JOINT.torso),
            (self.JOINT.right_shoulder_yaw, self.JOINT.right_shoulder_roll),
            (self.JOINT.right_shoulder_yaw, self.JOINT.right_shoulder_pitch),
            (self.JOINT.right_shoulder_yaw, self.JOINT.right_elbow),
            # right elbow
            (self.JOINT.right_elbow, self.JOINT.right_shoulder_roll),
            (self.JOINT.right_elbow, self.JOINT.right_shoulder_pitch),
            (self.JOINT.right_elbow, self.JOINT.right_shoulder_yaw),
        ]

    @property
    def connection_colors(self):
        return {
            "torso_ankle": "#FF0000",  # Red - Torso to ankle
            "torso_hip": "#FF6600",  # Orange - Torso to hip
            "torso_shoulder": "#0066FF",  # Blue - Torso to shoulder
            "torso_elbow": "#0033CC",  # Dark blue - Torso to elbow
            "torso_other": "#666666",  # Gray - Other torso connections
            "hip_internal": "#00CC00",  # Green - Hip internal connections
            "hip_knee": "#00AA00",  # Dark green - Hip to knee
            "knee_ankle": "#008800",  # Darker green - Knee to ankle
            "leg_other": "#004400",  # Very dark green - Other leg connections
            "shoulder_internal": "#00CCCC",  # Cyan - Shoulder internal connections
            "shoulder_elbow": "#0099AA",  # Teal - Shoulder to elbow
            "arm_other": "#006666",  # Dark teal - Other arm connections
            "cross_body": "#333333",  # Dark gray - Cross body connections
        }

    def get_joint_name(self, joint_id):
        """Convert joint ID to readable name using enum name"""
        try:
            # Get the enum member from the joint_id value
            joint_enum = self.JOINT(joint_id)
            # Return the enum name directly
            return joint_enum.name
        except ValueError:
            # Fallback for invalid joint IDs
            return f"Joint_{joint_id}"

    def get_robot_layout_positions(self):
        """Define positions to create a robot-like symmetric layout"""
        positions = {}

        # Torso at center
        positions[self.JOINT.torso] = (0, 0)

        # Left arm (from torso perspective)
        positions[self.JOINT.left_shoulder_pitch] = (-1, 0.6)
        positions[self.JOINT.left_shoulder_roll] = (-0.7, 0.3)
        positions[self.JOINT.left_shoulder_yaw] = (-1, 0)
        positions[self.JOINT.left_elbow] = (-1.5, 0.3)

        # Right arm (symmetric)
        positions[self.JOINT.right_shoulder_pitch] = (1, 0.6)
        positions[self.JOINT.right_shoulder_roll] = (0.7, 0.3)
        positions[self.JOINT.right_shoulder_yaw] = (1, 0)
        positions[self.JOINT.right_elbow] = (1.5, 0.3)

        # Left leg
        positions[self.JOINT.left_hip_yaw] = (-0.5, -0.5)
        positions[self.JOINT.left_hip_roll] = (-0.7, -1)
        positions[self.JOINT.left_hip_pitch] = (-0.3, -1)
        positions[self.JOINT.left_knee] = (-0.5, -2)
        positions[self.JOINT.left_ankle] = (-0.5, -2.5)

        # Right leg (symmetric)
        positions[self.JOINT.right_hip_yaw] = (0.5, -0.5)
        positions[self.JOINT.right_hip_roll] = (0.7, -1)
        positions[self.JOINT.right_hip_pitch] = (0.3, -1)
        positions[self.JOINT.right_knee] = (0.5, -2)
        positions[self.JOINT.right_ankle] = (0.5, -2.5)

        return positions

    def get_connection_type(self, joint1_id, joint2_id):
        # Ensure consistent ordering for lookup
        j1, j2 = sorted([joint1_id, joint2_id])

        # Torso connections (highest priority)
        if j1 == self.JOINT.torso or j2 == self.JOINT.torso:
            other_joint = j1 if j2 == self.JOINT.torso else j2

            # Torso to ankle
            if other_joint in [self.JOINT.left_ankle, self.JOINT.right_ankle]:
                return "torso_ankle"
            # Torso to hip
            elif other_joint in [
                self.JOINT.left_hip_yaw,
                self.JOINT.left_hip_roll,
                self.JOINT.left_hip_pitch,
                self.JOINT.right_hip_yaw,
                self.JOINT.right_hip_roll,
                self.JOINT.right_hip_pitch,
            ]:
                return "torso_hip"
            # Torso to shoulder
            elif other_joint in [
                self.JOINT.left_shoulder_pitch,
                self.JOINT.left_shoulder_roll,
                self.JOINT.left_shoulder_yaw,
                self.JOINT.right_shoulder_pitch,
                self.JOINT.right_shoulder_roll,
                self.JOINT.right_shoulder_yaw,
            ]:
                return "torso_shoulder"
            # Torso to elbow
            elif other_joint in [self.JOINT.left_elbow, self.JOINT.right_elbow]:
                return "torso_elbow"
            else:
                return "torso_other"

        # Left leg connections
        left_leg_joints = [
            self.JOINT.left_hip_yaw,
            self.JOINT.left_hip_roll,
            self.JOINT.left_hip_pitch,
            self.JOINT.left_knee,
            self.JOINT.left_ankle,
        ]
        if j1 in left_leg_joints and j2 in left_leg_joints:
            # Hip internal connections
            left_hip_joints = [
                self.JOINT.left_hip_yaw,
                self.JOINT.left_hip_roll,
                self.JOINT.left_hip_pitch,
            ]
            if j1 in left_hip_joints and j2 in left_hip_joints:
                return "hip_internal"
            # Hip to knee
            elif (j1 in left_hip_joints and j2 == self.JOINT.left_knee) or (
                j1 == self.JOINT.left_knee and j2 in left_hip_joints
            ):
                return "hip_knee"
            # Knee to ankle
            elif (j1 == self.JOINT.left_knee and j2 == self.JOINT.left_ankle) or (
                j1 == self.JOINT.left_ankle and j2 == self.JOINT.left_knee
            ):
                return "knee_ankle"
            else:
                return "leg_other"

        # Right leg connections
        right_leg_joints = [
            self.JOINT.right_hip_yaw,
            self.JOINT.right_hip_roll,
            self.JOINT.right_hip_pitch,
            self.JOINT.right_knee,
            self.JOINT.right_ankle,
        ]
        if j1 in right_leg_joints and j2 in right_leg_joints:
            # Hip internal connections
            right_hip_joints = [
                self.JOINT.right_hip_yaw,
                self.JOINT.right_hip_roll,
                self.JOINT.right_hip_pitch,
            ]
            if j1 in right_hip_joints and j2 in right_hip_joints:
                return "hip_internal"
            # Hip to knee
            elif (j1 in right_hip_joints and j2 == self.JOINT.right_knee) or (
                j1 == self.JOINT.right_knee and j2 in right_hip_joints
            ):
                return "hip_knee"
            # Knee to ankle
            elif (j1 == self.JOINT.right_knee and j2 == self.JOINT.right_ankle) or (
                j1 == self.JOINT.right_ankle and j2 == self.JOINT.right_knee
            ):
                return "knee_ankle"
            else:
                return "leg_other"

        # Left arm connections
        left_arm_joints = [
            self.JOINT.left_shoulder_pitch,
            self.JOINT.left_shoulder_roll,
            self.JOINT.left_shoulder_yaw,
            self.JOINT.left_elbow,
        ]
        if j1 in left_arm_joints and j2 in left_arm_joints:
            # Shoulder internal connections
            left_shoulder_joints = [
                self.JOINT.left_shoulder_pitch,
                self.JOINT.left_shoulder_roll,
                self.JOINT.left_shoulder_yaw,
            ]
            if j1 in left_shoulder_joints and j2 in left_shoulder_joints:
                return "shoulder_internal"
            # Shoulder to elbow
            elif (j1 in left_shoulder_joints and j2 == self.JOINT.left_elbow) or (
                j1 == self.JOINT.left_elbow and j2 in left_shoulder_joints
            ):
                return "shoulder_elbow"
            else:
                return "arm_other"

        # Right arm connections
        right_arm_joints = [
            self.JOINT.right_shoulder_pitch,
            self.JOINT.right_shoulder_roll,
            self.JOINT.right_shoulder_yaw,
            self.JOINT.right_elbow,
        ]
        if j1 in right_arm_joints and j2 in right_arm_joints:
            # Shoulder internal connections
            right_shoulder_joints = [
                self.JOINT.right_shoulder_pitch,
                self.JOINT.right_shoulder_roll,
                self.JOINT.right_shoulder_yaw,
            ]
            if j1 in right_shoulder_joints and j2 in right_shoulder_joints:
                return "shoulder_internal"
            # Shoulder to elbow
            elif (j1 in right_shoulder_joints and j2 == self.JOINT.right_elbow) or (
                j1 == self.JOINT.right_elbow and j2 in right_shoulder_joints
            ):
                return "shoulder_elbow"
            else:
                return "arm_other"

        # Cross-body connections
        return "cross_body"
