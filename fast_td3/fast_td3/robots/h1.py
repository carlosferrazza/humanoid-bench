import enum
from .robot import Robot


class H1(Robot):
    """H1 Robot metadata and topology."""

    def __init__(self):
        super().__init__()
        self._connections = set()
        self._initialize_connections()

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

    def add_connection(self, joint1, joint2):
        """Add a bidirectional connection between two joints."""
        self._connections.add((joint1, joint2))
        self._connections.add((joint2, joint1))

    def _initialize_connections(self):
        """Initialize the default joint connections."""

        default_undirected_connections = [
            # Left leg chain and internal connections
            (self.JOINT.left_hip_yaw, self.JOINT.left_hip_roll),
            (self.JOINT.left_hip_yaw, self.JOINT.left_hip_pitch),
            (self.JOINT.left_hip_roll, self.JOINT.left_hip_pitch),
            (self.JOINT.left_hip_yaw, self.JOINT.left_knee),
            (self.JOINT.left_hip_roll, self.JOINT.left_knee),
            (self.JOINT.left_hip_pitch, self.JOINT.left_knee),
            (self.JOINT.left_knee, self.JOINT.left_ankle),
            # Right leg chain and internal connections
            (self.JOINT.right_hip_yaw, self.JOINT.right_hip_roll),
            (self.JOINT.right_hip_yaw, self.JOINT.right_hip_pitch),
            (self.JOINT.right_hip_roll, self.JOINT.right_hip_pitch),
            (self.JOINT.right_hip_yaw, self.JOINT.right_knee),
            (self.JOINT.right_hip_roll, self.JOINT.right_knee),
            (self.JOINT.right_hip_pitch, self.JOINT.right_knee),
            (self.JOINT.right_knee, self.JOINT.right_ankle),
            # Left arm chain and internal connections
            (self.JOINT.left_shoulder_pitch, self.JOINT.left_shoulder_roll),
            (self.JOINT.left_shoulder_pitch, self.JOINT.left_shoulder_yaw),
            (self.JOINT.left_shoulder_roll, self.JOINT.left_shoulder_yaw),
            (self.JOINT.left_shoulder_roll, self.JOINT.left_elbow),
            (self.JOINT.left_shoulder_pitch, self.JOINT.left_elbow),
            (self.JOINT.left_shoulder_roll, self.JOINT.left_elbow),
            (self.JOINT.left_shoulder_yaw, self.JOINT.left_elbow),
            # Right arm chain and internal connections
            (self.JOINT.right_shoulder_pitch, self.JOINT.right_shoulder_roll),
            (self.JOINT.right_shoulder_pitch, self.JOINT.right_shoulder_yaw),
            (self.JOINT.right_shoulder_roll, self.JOINT.right_shoulder_yaw),
            (self.JOINT.right_shoulder_pitch, self.JOINT.right_elbow),
            (self.JOINT.right_shoulder_roll, self.JOINT.right_elbow),
            (self.JOINT.right_shoulder_yaw, self.JOINT.right_elbow),
            # Torso connections to all joints
            (self.JOINT.torso, self.JOINT.left_hip_yaw),
            (self.JOINT.torso, self.JOINT.left_hip_roll),
            (self.JOINT.torso, self.JOINT.left_hip_pitch),
            (self.JOINT.torso, self.JOINT.left_knee),
            (self.JOINT.torso, self.JOINT.left_ankle),
            (self.JOINT.torso, self.JOINT.right_hip_yaw),
            (self.JOINT.torso, self.JOINT.right_hip_roll),
            (self.JOINT.torso, self.JOINT.right_hip_pitch),
            (self.JOINT.torso, self.JOINT.right_knee),
            (self.JOINT.torso, self.JOINT.right_ankle),
            (self.JOINT.torso, self.JOINT.left_shoulder_pitch),
            (self.JOINT.torso, self.JOINT.left_shoulder_roll),
            (self.JOINT.torso, self.JOINT.left_shoulder_yaw),
            (self.JOINT.torso, self.JOINT.left_elbow),
            (self.JOINT.torso, self.JOINT.right_shoulder_pitch),
            (self.JOINT.torso, self.JOINT.right_shoulder_roll),
            (self.JOINT.torso, self.JOINT.right_shoulder_yaw),
            (self.JOINT.torso, self.JOINT.right_elbow),
        ]
        # Add each undirected pair (bidirectional will be added automatically)
        for a, b in default_undirected_connections:
            self.add_connection(a, b)

    @property
    def num_edges(self):
        """Number of edges in the robot graph."""
        return len(self.joint_connections)
    
    @property
    def num_joints(self):
        """Number of joints in the robot graph."""
        return len(self.JOINT)

    @property
    def joint_connections(self):
        """List of directed joint-to-joint edges (no object)."""
        return sorted(self._connections)



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
        """Convert a node id to readable name."""
        try:
            return self.JOINT(joint_id).name
        except ValueError:
            return f"node_{joint_id}"

    def get_robot_layout_positions(self):
        """Define positions to create a robot-like symmetric layout."""
        positions = {}
        positions[self.JOINT.torso] = (0, 0)  # Torso center

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
