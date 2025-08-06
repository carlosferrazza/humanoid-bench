import enum
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict

class NodeType(str, enum.Enum):
    JOINT = "joint"

class H1:
    joint_dict = {
        "left_hip_yaw": 0,
        "left_hip_roll": 1,
        "left_hip_pitch": 2,
        "left_knee": 3,
        "left_ankle": 4,
        "right_hip_yaw": 5,
        "right_hip_roll": 6,
        "right_hip_pitch": 7,
        "right_knee": 8,
        "right_ankle": 9,
        "torso": 10,
        "left_shoulder_pitch": 11,
        "left_shoulder_roll": 12,
        "left_shoulder_yaw": 13,
        "left_elbow": 14,
        "right_shoulder_pitch": 15,
        "right_shoulder_roll": 16,
        "right_shoulder_yaw": 17,
        "right_elbow": 18,
    }

    edge_list = [
        # Left leg
        (joint_dict["left_hip_yaw"], joint_dict["left_hip_roll"]),
        (joint_dict["left_hip_roll"], joint_dict["left_hip_pitch"]),
        (joint_dict["left_hip_pitch"], joint_dict["left_knee"]),
        (joint_dict["left_knee"], joint_dict["left_ankle"]),
        # Right leg
        (joint_dict["right_hip_yaw"], joint_dict["right_hip_roll"]),
        (joint_dict["right_hip_roll"], joint_dict["right_hip_pitch"]),
        (joint_dict["right_hip_pitch"], joint_dict["right_knee"]),
        (joint_dict["right_knee"], joint_dict["right_ankle"]),
        # Torso
        (joint_dict["torso"], joint_dict["left_hip_yaw"]),
        (joint_dict["torso"], joint_dict["right_hip_yaw"]),
        # Left arm
        (joint_dict["torso"], joint_dict["left_shoulder_pitch"]),
        (joint_dict["left_shoulder_pitch"], joint_dict["left_shoulder_roll"]),
        (joint_dict["left_shoulder_roll"], joint_dict["left_shoulder_yaw"]),
        (joint_dict["left_shoulder_yaw"], joint_dict["left_elbow"]),
        # Right arm
        (joint_dict["torso"], joint_dict["right_shoulder_pitch"]),
        (joint_dict["right_shoulder_pitch"], joint_dict["right_shoulder_roll"]),
        (joint_dict["right_shoulder_roll"], joint_dict["right_shoulder_yaw"]),
        (joint_dict["right_shoulder_yaw"], joint_dict["right_elbow"]),
    ]

    edge_type_encoding = [
        0,  # (left_hip_yaw, left_hip_roll)      - left_leg
        1,  # (left_hip_roll, left_hip_pitch)    - left_leg
        2,  # (left_hip_pitch, left_knee)        - left_leg
        3,  # (left_knee, left_ankle)            - left_leg
        0,  # (right_hip_yaw, right_hip_roll)    - right_leg
        1,  # (right_hip_roll, right_hip_pitch)  - right_leg
        2,  # (right_hip_pitch, right_knee)      - right_leg
        3,  # (right_knee, right_ankle)          - right_leg
        4,  # (torso, left_hip_yaw)              - torso_connection
        4,  # (torso, right_hip_yaw)             - torso_connection
        5,  # (torso, left_shoulder_pitch)      - left_arm
        6,  # (left_shoulder_pitch, left_shoulder_roll) - left_arm
        7,  # (left_shoulder_roll, left_shoulder_yaw)   - left_arm
        8,  # (left_shoulder_yaw, left_elbow)            - left_arm
        5,  # (torso, right_shoulder_pitch)     - right_arm
        6,  # (right_shoulder_pitch, right_shoulder_roll) - right_arm
        7,  # (right_shoulder_roll, right_shoulder_yaw)   - right_arm
        8,  # (right_shoulder_yaw, right_elbow)            - right_arm
    ]

    node_type_encoding = [
        0,  # left_hip_yaw
        1,  # left_hip_roll,
        2,  # left_hip_pitch
        3,  # left_knee
        4,  # left_ankle
        0,  # right_hip_yaw
        1,  # right_hip_roll
        2,  # right_hip_pitch
        3,  # right_knee
        4,  # right_ankle
        5,  # torso
        6,  # left_shoulder_pitch
        7,  # left_shoulder_roll
        8,  # left_shoulder_yaw
        9,  # left_elbow
        6,  # right_shoulder_pitch
        7,  # right_shoulder_roll
        8,  # right_shoulder_yaw
        9,  # right_elbow
    ]

    num_nodes = len(joint_dict)
    num_edges = len(edge_list)

    JOINT_TYPE_MAP = {
        "left_hip_yaw": "hip",
        "left_hip_roll": "hip",
        "left_hip_pitch": "hip",
        "right_hip_yaw": "hip",
        "right_hip_roll": "hip",
        "right_hip_pitch": "hip",
        "left_knee": "knee",
        "right_knee": "knee",
        "left_ankle": "ankle",
        "right_ankle": "ankle",
        "torso": "torso",
        "left_shoulder_pitch": "shoulder",
        "left_shoulder_roll": "shoulder",
        "left_shoulder_yaw": "shoulder",
        "left_elbow": "elbow",
        "right_shoulder_pitch": "shoulder",
        "right_shoulder_roll": "shoulder",
        "right_shoulder_yaw": "shoulder",
        "right_elbow": "elbow",
    }

    node_type = NodeType

    def create_joint_graph(self, batch_edge_index, batch_edge_attr,  joint_positions) -> HeteroData:
        hetero_data = HeteroData()
        hetero_data[self.node_type.JOINT].x = joint_positions
        hetero_data[self.node_type.JOINT, 'connects', self.node_type.JOINT].edge_index = batch_edge_index
        hetero_data[self.node_type.JOINT, 'connects', self.node_type.JOINT].edge_attr = batch_edge_attr
        
        return hetero_data

h1 = H1()


if __name__ == "__main__":
    h1 = H1()

    batch_size = 4
    num_nodes = 19

    x = torch.randn(batch_size*num_nodes, 3)
    edge_index = torch.rand(batch_size*18, 1) 
    edge_attr = torch.rand(batch_size*18, 1)   

    graph = h1.create_joint_graph(edge_index, edge_attr, x)
    print(graph)