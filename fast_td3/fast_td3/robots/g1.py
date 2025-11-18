import enum
from .robot import Robot


class G1(Robot):
    """G1 Robot metadata and topology."""

    def __init__(self):
        super().__init__()
        self._connections = set()
        self._initialize_connections()

    class JOINT(enum.IntEnum):
        # Left leg
        left_hip_pitch = 0
        left_hip_roll = 1
        left_hip_yaw = 2
        left_knee = 3
        left_ankle_pitch = 4
        left_ankle_roll = 5

        # Right leg
        right_hip_pitch = 6
        right_hip_roll = 7
        right_hip_yaw = 8
        right_knee = 9
        right_ankle_pitch = 10
        right_ankle_roll = 11

        # Torso
        torso = 12

        # Left arm
        left_shoulder_pitch = 13
        left_shoulder_roll = 14
        left_shoulder_yaw = 15
        left_elbow_pitch = 16
        left_elbow_roll = 17
        left_zero = 18
        left_one = 19
        left_two = 20
        left_three = 21
        left_four = 22
        left_five = 23
        left_six = 24

        # Right arm
        right_shoulder_pitch = 25
        right_shoulder_roll = 26
        right_shoulder_yaw = 27
        right_elbow_pitch = 28
        right_elbow_roll = 29
        right_zero = 30
        right_one = 31
        right_two = 32
        right_three = 33
        right_four = 34
        right_five = 35
        right_six = 36

    def add_connection(self, joint1, joint2):
        """Add a bidirectional connection between two joints."""
        self._connections.add((joint1, joint2))
        self._connections.add((joint2, joint1))

    def _initialize_connections(self):
        """Initialize the default joint connections."""

        default_undirected_connections = [
            # Left leg chain and internal connections
            (self.JOINT.left_hip_pitch, self.JOINT.left_hip_roll),
            (self.JOINT.left_hip_pitch, self.JOINT.left_hip_yaw),
            (self.JOINT.left_hip_roll, self.JOINT.left_hip_yaw),
            (self.JOINT.left_hip_pitch, self.JOINT.left_knee),
            (self.JOINT.left_hip_roll, self.JOINT.left_knee),
            (self.JOINT.left_hip_yaw, self.JOINT.left_knee),
            (self.JOINT.left_knee, self.JOINT.left_ankle_pitch),
            (self.JOINT.left_ankle_pitch, self.JOINT.left_ankle_roll),
            
            # Right leg chain and internal connections
            (self.JOINT.right_hip_pitch, self.JOINT.right_hip_roll),
            (self.JOINT.right_hip_pitch, self.JOINT.right_hip_yaw),
            (self.JOINT.right_hip_roll, self.JOINT.right_hip_yaw),
            (self.JOINT.right_hip_pitch, self.JOINT.right_knee),
            (self.JOINT.right_hip_roll, self.JOINT.right_knee),
            (self.JOINT.right_hip_yaw, self.JOINT.right_knee),
            (self.JOINT.right_knee, self.JOINT.right_ankle_pitch),
            (self.JOINT.right_ankle_pitch, self.JOINT.right_ankle_roll),
            
            # Left arm chain and internal connections
            (self.JOINT.left_shoulder_pitch, self.JOINT.left_shoulder_roll),
            (self.JOINT.left_shoulder_pitch, self.JOINT.left_shoulder_yaw),
            (self.JOINT.left_shoulder_roll, self.JOINT.left_shoulder_yaw),
            (self.JOINT.left_shoulder_yaw, self.JOINT.left_elbow_pitch),
            (self.JOINT.left_shoulder_pitch, self.JOINT.left_elbow_pitch),
            (self.JOINT.left_shoulder_roll, self.JOINT.left_elbow_pitch),
            (self.JOINT.left_elbow_pitch, self.JOINT.left_elbow_roll),
            (self.JOINT.left_elbow_roll, self.JOINT.left_zero),
            (self.JOINT.left_zero, self.JOINT.left_one),
            (self.JOINT.left_one, self.JOINT.left_two),
            (self.JOINT.left_zero, self.JOINT.left_three),
            (self.JOINT.left_three, self.JOINT.left_four),
            (self.JOINT.left_three, self.JOINT.left_five),
            (self.JOINT.left_five, self.JOINT.left_six),
            
            # Right arm chain and internal connections
            (self.JOINT.right_shoulder_pitch, self.JOINT.right_shoulder_roll),
            (self.JOINT.right_shoulder_pitch, self.JOINT.right_shoulder_yaw),
            (self.JOINT.right_shoulder_roll, self.JOINT.right_shoulder_yaw),
            (self.JOINT.right_shoulder_yaw, self.JOINT.right_elbow_pitch),
            (self.JOINT.right_shoulder_pitch, self.JOINT.right_elbow_pitch),
            (self.JOINT.right_shoulder_roll, self.JOINT.right_elbow_pitch),
            (self.JOINT.right_elbow_pitch, self.JOINT.right_elbow_roll),
            (self.JOINT.right_elbow_roll, self.JOINT.right_zero),
            (self.JOINT.right_zero, self.JOINT.right_one),
            (self.JOINT.right_one, self.JOINT.right_two),
            (self.JOINT.right_zero, self.JOINT.right_three),
            (self.JOINT.right_three, self.JOINT.right_four),
            (self.JOINT.right_three, self.JOINT.right_five),
            (self.JOINT.right_five, self.JOINT.right_six),
            
            # Torso connections to all joints
            (self.JOINT.torso, self.JOINT.left_hip_pitch),
            (self.JOINT.torso, self.JOINT.left_hip_roll),
            (self.JOINT.torso, self.JOINT.left_hip_yaw),
            (self.JOINT.torso, self.JOINT.left_knee),
            (self.JOINT.torso, self.JOINT.left_ankle_pitch),
            (self.JOINT.torso, self.JOINT.left_ankle_roll),
            (self.JOINT.torso, self.JOINT.right_hip_pitch),
            (self.JOINT.torso, self.JOINT.right_hip_roll),
            (self.JOINT.torso, self.JOINT.right_hip_yaw),
            (self.JOINT.torso, self.JOINT.right_knee),
            (self.JOINT.torso, self.JOINT.right_ankle_pitch),
            (self.JOINT.torso, self.JOINT.right_ankle_roll),
            (self.JOINT.torso, self.JOINT.left_shoulder_pitch),
            (self.JOINT.torso, self.JOINT.left_shoulder_roll),
            (self.JOINT.torso, self.JOINT.left_shoulder_yaw),
            (self.JOINT.torso, self.JOINT.left_elbow_pitch),
            (self.JOINT.torso, self.JOINT.left_elbow_roll),
            (self.JOINT.torso, self.JOINT.right_shoulder_pitch),
            (self.JOINT.torso, self.JOINT.right_shoulder_roll),
            (self.JOINT.torso, self.JOINT.right_shoulder_yaw),
            (self.JOINT.torso, self.JOINT.right_elbow_pitch),
            (self.JOINT.torso, self.JOINT.right_elbow_roll),
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
            "ankle_internal": "#006600",  # Very dark green - Ankle internal
            "leg_other": "#004400",  # Very dark green - Other leg connections
            "shoulder_internal": "#00CCCC",  # Cyan - Shoulder internal connections
            "shoulder_elbow": "#0099AA",  # Teal - Shoulder to elbow
            "elbow_internal": "#007788",  # Dark teal - Elbow internal
            "hand_internal": "#005555",  # Darker teal - Hand/finger connections
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
        positions[self.JOINT.left_elbow_pitch] = (-1.5, 0.3)
        positions[self.JOINT.left_elbow_roll] = (-1.8, 0.3)
        positions[self.JOINT.left_zero] = (-2.1, 0.5)
        positions[self.JOINT.left_one] = (-2.3, 0.7)
        positions[self.JOINT.left_two] = (-2.5, 0.9)
        positions[self.JOINT.left_three] = (-2.3, 0.3)
        positions[self.JOINT.left_four] = (-2.5, 0.4)
        positions[self.JOINT.left_five] = (-2.5, 0.1)
        positions[self.JOINT.left_six] = (-2.7, 0.0)

        # Right arm (symmetric)
        positions[self.JOINT.right_shoulder_pitch] = (1, 0.6)
        positions[self.JOINT.right_shoulder_roll] = (0.7, 0.3)
        positions[self.JOINT.right_shoulder_yaw] = (1, 0)
        positions[self.JOINT.right_elbow_pitch] = (1.5, 0.3)
        positions[self.JOINT.right_elbow_roll] = (1.8, 0.3)
        positions[self.JOINT.right_zero] = (2.1, 0.5)
        positions[self.JOINT.right_one] = (2.3, 0.7)
        positions[self.JOINT.right_two] = (2.5, 0.9)
        positions[self.JOINT.right_three] = (2.3, 0.3)
        positions[self.JOINT.right_four] = (2.5, 0.4)
        positions[self.JOINT.right_five] = (2.5, 0.1)
        positions[self.JOINT.right_six] = (2.7, 0.0)

        # Left leg
        positions[self.JOINT.left_hip_pitch] = (-0.5, -0.5)
        positions[self.JOINT.left_hip_roll] = (-0.7, -1)
        positions[self.JOINT.left_hip_yaw] = (-0.3, -1)
        positions[self.JOINT.left_knee] = (-0.5, -2)
        positions[self.JOINT.left_ankle_pitch] = (-0.5, -2.5)
        positions[self.JOINT.left_ankle_roll] = (-0.5, -2.8)

        # Right leg (symmetric)
        positions[self.JOINT.right_hip_pitch] = (0.5, -0.5)
        positions[self.JOINT.right_hip_roll] = (0.7, -1)
        positions[self.JOINT.right_hip_yaw] = (0.3, -1)
        positions[self.JOINT.right_knee] = (0.5, -2)
        positions[self.JOINT.right_ankle_pitch] = (0.5, -2.5)
        positions[self.JOINT.right_ankle_roll] = (0.5, -2.8)

        return positions

    def get_connection_type(self, joint1_id, joint2_id):
        # Ensure consistent ordering for lookup
        j1, j2 = sorted([joint1_id, joint2_id])

        # Torso connections (highest priority)
        if j1 == self.JOINT.torso or j2 == self.JOINT.torso:
            other_joint = j1 if j2 == self.JOINT.torso else j2

            # Torso to ankle
            if other_joint in [self.JOINT.left_ankle_pitch, self.JOINT.left_ankle_roll,
                             self.JOINT.right_ankle_pitch, self.JOINT.right_ankle_roll]:
                return "torso_ankle"
            # Torso to hip
            elif other_joint in [
                self.JOINT.left_hip_pitch, self.JOINT.left_hip_roll, self.JOINT.left_hip_yaw,
                self.JOINT.right_hip_pitch, self.JOINT.right_hip_roll, self.JOINT.right_hip_yaw,
            ]:
                return "torso_hip"
            # Torso to shoulder
            elif other_joint in [
                self.JOINT.left_shoulder_pitch, self.JOINT.left_shoulder_roll, self.JOINT.left_shoulder_yaw,
                self.JOINT.right_shoulder_pitch, self.JOINT.right_shoulder_roll, self.JOINT.right_shoulder_yaw,
            ]:
                return "torso_shoulder"
            # Torso to elbow
            elif other_joint in [self.JOINT.left_elbow_pitch, self.JOINT.left_elbow_roll,
                               self.JOINT.right_elbow_pitch, self.JOINT.right_elbow_roll]:
                return "torso_elbow"
            else:
                return "torso_other"

        # Left leg connections
        left_leg_joints = [
            self.JOINT.left_hip_pitch, self.JOINT.left_hip_roll, self.JOINT.left_hip_yaw,
            self.JOINT.left_knee, self.JOINT.left_ankle_pitch, self.JOINT.left_ankle_roll,
        ]
        if j1 in left_leg_joints and j2 in left_leg_joints:
            # Hip internal connections
            left_hip_joints = [self.JOINT.left_hip_pitch, self.JOINT.left_hip_roll, self.JOINT.left_hip_yaw]
            if j1 in left_hip_joints and j2 in left_hip_joints:
                return "hip_internal"
            # Ankle internal connections
            left_ankle_joints = [self.JOINT.left_ankle_pitch, self.JOINT.left_ankle_roll]
            if j1 in left_ankle_joints and j2 in left_ankle_joints:
                return "ankle_internal"
            # Hip to knee
            elif (j1 in left_hip_joints and j2 == self.JOINT.left_knee) or (
                j1 == self.JOINT.left_knee and j2 in left_hip_joints
            ):
                return "hip_knee"
            # Knee to ankle
            elif (j1 == self.JOINT.left_knee and j2 in left_ankle_joints) or (
                j1 in left_ankle_joints and j2 == self.JOINT.left_knee
            ):
                return "knee_ankle"
            else:
                return "leg_other"

        # Right leg connections
        right_leg_joints = [
            self.JOINT.right_hip_pitch, self.JOINT.right_hip_roll, self.JOINT.right_hip_yaw,
            self.JOINT.right_knee, self.JOINT.right_ankle_pitch, self.JOINT.right_ankle_roll,
        ]
        if j1 in right_leg_joints and j2 in right_leg_joints:
            # Hip internal connections
            right_hip_joints = [self.JOINT.right_hip_pitch, self.JOINT.right_hip_roll, self.JOINT.right_hip_yaw]
            if j1 in right_hip_joints and j2 in right_hip_joints:
                return "hip_internal"
            # Ankle internal connections
            right_ankle_joints = [self.JOINT.right_ankle_pitch, self.JOINT.right_ankle_roll]
            if j1 in right_ankle_joints and j2 in right_ankle_joints:
                return "ankle_internal"
            # Hip to knee
            elif (j1 in right_hip_joints and j2 == self.JOINT.right_knee) or (
                j1 == self.JOINT.right_knee and j2 in right_hip_joints
            ):
                return "hip_knee"
            # Knee to ankle
            elif (j1 == self.JOINT.right_knee and j2 in right_ankle_joints) or (
                j1 in right_ankle_joints and j2 == self.JOINT.right_knee
            ):
                return "knee_ankle"
            else:
                return "leg_other"

        # Left arm connections
        left_arm_joints = [
            self.JOINT.left_shoulder_pitch, self.JOINT.left_shoulder_roll, self.JOINT.left_shoulder_yaw,
            self.JOINT.left_elbow_pitch, self.JOINT.left_elbow_roll,
            self.JOINT.left_zero, self.JOINT.left_one, self.JOINT.left_two,
            self.JOINT.left_three, self.JOINT.left_four, self.JOINT.left_five, self.JOINT.left_six,
        ]
        if j1 in left_arm_joints and j2 in left_arm_joints:
            # Shoulder internal connections
            left_shoulder_joints = [
                self.JOINT.left_shoulder_pitch, self.JOINT.left_shoulder_roll, self.JOINT.left_shoulder_yaw,
            ]
            if j1 in left_shoulder_joints and j2 in left_shoulder_joints:
                return "shoulder_internal"
            # Elbow internal connections
            left_elbow_joints = [self.JOINT.left_elbow_pitch, self.JOINT.left_elbow_roll]
            if j1 in left_elbow_joints and j2 in left_elbow_joints:
                return "elbow_internal"
            # Hand/finger connections
            left_hand_joints = [
                self.JOINT.left_zero, self.JOINT.left_one, self.JOINT.left_two,
                self.JOINT.left_three, self.JOINT.left_four, self.JOINT.left_five, self.JOINT.left_six,
            ]
            if j1 in left_hand_joints and j2 in left_hand_joints:
                return "hand_internal"
            # Shoulder to elbow
            elif (j1 in left_shoulder_joints and j2 in left_elbow_joints) or (
                j1 in left_elbow_joints and j2 in left_shoulder_joints
            ):
                return "shoulder_elbow"
            else:
                return "arm_other"

        # Right arm connections
        right_arm_joints = [
            self.JOINT.right_shoulder_pitch, self.JOINT.right_shoulder_roll, self.JOINT.right_shoulder_yaw,
            self.JOINT.right_elbow_pitch, self.JOINT.right_elbow_roll,
            self.JOINT.right_zero, self.JOINT.right_one, self.JOINT.right_two,
            self.JOINT.right_three, self.JOINT.right_four, self.JOINT.right_five, self.JOINT.right_six,
        ]
        if j1 in right_arm_joints and j2 in right_arm_joints:
            # Shoulder internal connections
            right_shoulder_joints = [
                self.JOINT.right_shoulder_pitch, self.JOINT.right_shoulder_roll, self.JOINT.right_shoulder_yaw,
            ]
            if j1 in right_shoulder_joints and j2 in right_shoulder_joints:
                return "shoulder_internal"
            # Elbow internal connections
            right_elbow_joints = [self.JOINT.right_elbow_pitch, self.JOINT.right_elbow_roll]
            if j1 in right_elbow_joints and j2 in right_elbow_joints:
                return "elbow_internal"
            # Hand/finger connections
            right_hand_joints = [
                self.JOINT.right_zero, self.JOINT.right_one, self.JOINT.right_two,
                self.JOINT.right_three, self.JOINT.right_four, self.JOINT.right_five, self.JOINT.right_six,
            ]
            if j1 in right_hand_joints and j2 in right_hand_joints:
                return "hand_internal"
            # Shoulder to elbow
            elif (j1 in right_shoulder_joints and j2 in right_elbow_joints) or (
                j1 in right_elbow_joints and j2 in right_shoulder_joints
            ):
                return "shoulder_elbow"
            else:
                return "arm_other"

        # Cross-body connections
        return "cross_body"