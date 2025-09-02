import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation as R

class H1:
    joint_dict = {
        "free_base": 0,  # 6-DOF free joint (3 translation + 3 rotation)
        "left_hip_yaw": 1,
        "left_hip_roll": 2,
        "left_hip_pitch": 3,
        "left_knee": 4,
        "left_ankle": 5,
        "right_hip_yaw": 6,
        "right_hip_roll": 7,
        "right_hip_pitch": 8,
        "right_knee": 9,
        "right_ankle": 10,
        "torso": 11,
        "left_shoulder_pitch": 12,
        "left_shoulder_roll": 13,
        "left_shoulder_yaw": 14,
        "left_elbow": 15,
        "right_shoulder_pitch": 16,
        "right_shoulder_roll": 17,
        "right_shoulder_yaw": 18,
        "right_elbow": 19,
    }

    edge_list = [
        # free_base (pelvis) connections - connects to all immediate children
        (joint_dict["free_base"], joint_dict["left_hip_yaw"]),
        (joint_dict["free_base"], joint_dict["right_hip_yaw"]),
        (joint_dict["free_base"], joint_dict["torso"]),
        
        # left leg chain
        (joint_dict["left_hip_yaw"], joint_dict["free_base"]),
        (joint_dict["left_hip_yaw"], joint_dict["left_hip_roll"]),
        (joint_dict["left_hip_roll"], joint_dict["left_hip_yaw"]),
        (joint_dict["left_hip_roll"], joint_dict["left_hip_pitch"]),
        (joint_dict["left_hip_pitch"], joint_dict["left_hip_roll"]),
        (joint_dict["left_hip_pitch"], joint_dict["left_knee"]),
        (joint_dict["left_knee"], joint_dict["left_hip_pitch"]),
        (joint_dict["left_knee"], joint_dict["left_ankle"]),
        (joint_dict["left_ankle"], joint_dict["left_knee"]),
        
        # right leg chain
        (joint_dict["right_hip_yaw"], joint_dict["free_base"]),
        (joint_dict["right_hip_yaw"], joint_dict["right_hip_roll"]),
        (joint_dict["right_hip_roll"], joint_dict["right_hip_yaw"]),
        (joint_dict["right_hip_roll"], joint_dict["right_hip_pitch"]),
        (joint_dict["right_hip_pitch"], joint_dict["right_hip_roll"]),
        (joint_dict["right_hip_pitch"], joint_dict["right_knee"]),
        (joint_dict["right_knee"], joint_dict["right_hip_pitch"]),
        (joint_dict["right_knee"], joint_dict["right_ankle"]),
        (joint_dict["right_ankle"], joint_dict["right_knee"]),
        
        # torso connections
        (joint_dict["torso"], joint_dict["free_base"]),
        (joint_dict["torso"], joint_dict["left_shoulder_pitch"]),
        (joint_dict["torso"], joint_dict["right_shoulder_pitch"]),
        
        # left arm chain
        (joint_dict["left_shoulder_pitch"], joint_dict["torso"]),
        (joint_dict["left_shoulder_pitch"], joint_dict["left_shoulder_roll"]),
        (joint_dict["left_shoulder_roll"], joint_dict["left_shoulder_pitch"]),
        (joint_dict["left_shoulder_roll"], joint_dict["left_shoulder_yaw"]),
        (joint_dict["left_shoulder_yaw"], joint_dict["left_shoulder_roll"]),
        (joint_dict["left_shoulder_yaw"], joint_dict["left_elbow"]),
        (joint_dict["left_elbow"], joint_dict["left_shoulder_yaw"]),
        
        # right arm chain
        (joint_dict["right_shoulder_pitch"], joint_dict["torso"]),
        (joint_dict["right_shoulder_pitch"], joint_dict["right_shoulder_roll"]),
        (joint_dict["right_shoulder_roll"], joint_dict["right_shoulder_pitch"]),
        (joint_dict["right_shoulder_roll"], joint_dict["right_shoulder_yaw"]),
        (joint_dict["right_shoulder_yaw"], joint_dict["right_shoulder_roll"]),
        (joint_dict["right_shoulder_yaw"], joint_dict["right_elbow"]),
        (joint_dict["right_elbow"], joint_dict["right_shoulder_yaw"]),
    ]

    node_type_encoding = [
        10, # free_base (new type for 6-DOF base)
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

    # Define body tree structure and joint order
    # Only include main bodies for simplicity; you can extend to every link
    body_tree = {
        "pelvis": {
            "pos": np.array([0,0,0]), "quat": None, "joint": "free_base", "children": [
                {"name": "left_hip_yaw_link", "pos": np.array([0, 0.0875, -0.1742]), "joint": "left_hip_yaw", "children": [
                    {"name": "left_hip_roll_link", "pos": np.array([0.039468, 0, 0]), "joint": "left_hip_roll", "children": [
                        {"name": "left_hip_pitch_link", "pos": np.array([0,0.11536,0]), "joint": "left_hip_pitch", "children": [
                            {"name": "left_knee_link", "pos": np.array([0,0,-0.4]), "joint": "left_knee", "children": [
                                {"name": "left_ankle_link", "pos": np.array([0,0,-0.4]), "joint": "left_ankle", "children": []}
                            ]}
                        ]}
                    ]}
                ]},
                {"name": "right_hip_yaw_link", "pos": np.array([0, -0.0875, -0.1742]), "joint": "right_hip_yaw", "children": [
                    {"name": "right_hip_roll_link", "pos": np.array([0.039468,0,0]), "joint": "right_hip_roll", "children": [
                        {"name": "right_hip_pitch_link", "pos": np.array([0,-0.11536,0]), "joint": "right_hip_pitch", "children": [
                            {"name": "right_knee_link", "pos": np.array([0,0,-0.4]), "joint": "right_knee", "children": [
                                {"name": "right_ankle_link", "pos": np.array([0,0,-0.4]), "joint": "right_ankle", "children": []}
                            ]}
                        ]}
                    ]}
                ]},
                {"name": "torso_link", "pos": np.array([0,0,0]), "joint": "torso", "children": [
                    {"name": "left_shoulder_pitch_link", "pos": np.array([0.0055,0.15535,0.42999]), "joint": "left_shoulder_pitch", "children": [
                        {"name": "left_shoulder_roll_link", "pos": np.array([-0.0055,0.0565,-0.0165]), "joint": "left_shoulder_roll", "children": [
                            {"name": "left_shoulder_yaw_link", "pos": np.array([0,0,-0.1343]), "joint": "left_shoulder_yaw", "children": [
                                {"name": "left_elbow_link", "pos": np.array([0.0185,0,-0.198]), "joint": "left_elbow", "children": []}
                            ]}
                        ]}
                    ]},
                    {"name": "right_shoulder_pitch_link", "pos": np.array([0.0055,-0.15535,0.42999]), "joint": "right_shoulder_pitch", "children": [
                        {"name": "right_shoulder_roll_link", "pos": np.array([-0.0055,-0.0565,-0.0165]), "joint": "right_shoulder_roll", "children": [
                            {"name": "right_shoulder_yaw_link", "pos": np.array([0,0,-0.1343]), "joint": "right_shoulder_yaw", "children": [
                                {"name": "right_elbow_link", "pos": np.array([0.0185,0,-0.198]), "joint": "right_elbow", "children": []}
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    }

    # Joint order in qpos (matching your FieldIndexer)
    joint_indices = {
        "free_base": slice(0,7),   # 0-6: 3 pos + 4 quat
        "left_hip_yaw": 7,
        "left_hip_roll": 8,
        "left_hip_pitch": 9,
        "left_knee": 10,
        "left_ankle": 11,
        "right_hip_yaw": 12,
        "right_hip_roll": 13,
        "right_hip_pitch": 14,
        "right_knee": 15,
        "right_ankle": 16,
        "torso": 17,
        "left_shoulder_pitch": 18,
        "left_shoulder_roll": 19,
        "left_shoulder_yaw": 20,
        "left_elbow": 21,
        "right_shoulder_pitch": 22,
        "right_shoulder_roll": 23,
        "right_shoulder_yaw": 24,
        "right_elbow": 25
    }

    # Joint axes (from XML)
    joint_axes = {
        "left_hip_yaw": np.array([0,0,1]),
        "left_hip_roll": np.array([1,0,0]),
        "left_hip_pitch": np.array([0,1,0]),
        "left_knee": np.array([0,1,0]),
        "left_ankle": np.array([0,1,0]),
        "right_hip_yaw": np.array([0,0,1]),
        "right_hip_roll": np.array([1,0,0]),
        "right_hip_pitch": np.array([0,1,0]),
        "right_knee": np.array([0,1,0]),
        "right_ankle": np.array([0,1,0]),
        "torso": np.array([0,0,1]),
        "left_shoulder_pitch": np.array([0,1,0]),
        "left_shoulder_roll": np.array([1,0,0]),
        "left_shoulder_yaw": np.array([0,0,1]),
        "left_elbow": np.array([0,1,0]),
        "right_shoulder_pitch": np.array([0,1,0]),
        "right_shoulder_roll": np.array([1,0,0]),
        "right_shoulder_yaw": np.array([0,0,1]),
        "right_elbow": np.array([0,1,0])
    }

    def fk_joint_positions(self, body, qpos, parent_T=np.eye(4)):
        """
        Compute world-space joint anchor positions for the simplified tree.
        - Applies free base pose first.
        - Computes anchor = parent_T @ pos_offset (rotated) and records it.
        - Applies joint rotation about the anchor (no translation).
        - Recurse to children using the updated transform (origin moved to anchor).
        """
        # Start from parent transform
        T = parent_T.copy()

        # Identify this joint
        joint_name = body.get("joint", body.get("name", f"unnamed_{id(body)}"))
        pos_offset = body.get("pos", np.zeros(3))

        # If free base, set base pose first (pos + orientation)
        if joint_name == "free_base":
            base = qpos[self.joint_indices[joint_name]]
            base_pos = base[:3]
            base_quat_wxyz = base[3:7]  # [w,x,y,z]
            # scipy expects [x,y,z,w]
            base_quat_xyzw = [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]]
            T[:3, :3] = R.from_quat(base_quat_xyzw).as_matrix()
            T[:3, 3] = base_pos

        # Anchor world position: translate by offset expressed in parent's rotated frame
        anchor_world = T[:3, 3] + T[:3, :3] @ pos_offset

        # Record anchor position for this joint (strip optional _link suffix)
        joint_key = joint_name.replace("_link", "")
        joint_pos = {joint_key: anchor_world}

        # Rotate about anchor if revolute joint
        if joint_name in self.joint_axes:
            axis_local = self.joint_axes[joint_name]
            angle = qpos[self.joint_indices[joint_name]]
            R_joint = R.from_rotvec(axis_local * angle).as_matrix()
            T[:3, :3] = T[:3, :3] @ R_joint

        # Move origin to the anchor for children
        T[:3, 3] = anchor_world

        # Recurse
        for child in body.get("children", []):
            joint_pos.update(self.fk_joint_positions(child, qpos, T))

        return joint_pos


h1 = H1()

if __name__ == "__main__":
    h1 = H1()

    batch_size = 4
    num_nodes = 20  # Updated to include free_base

    x = torch.randn(batch_size * num_nodes, 3)
    edge_index = torch.rand(batch_size * len(h1.edge_list), 1)
    edge_attr = torch.rand(batch_size * len(h1.edge_list), 1)

    graph = h1.create_joint_graph(edge_index, edge_attr, x)
    print(graph)
