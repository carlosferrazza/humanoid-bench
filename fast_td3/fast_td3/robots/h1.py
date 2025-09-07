import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Ordered list of joint names for consistent tensor indexing
    joint_names_ordered = [
        "free_base", "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
        "torso", "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"
    ]

    edge_list = [
        # left hip yaw
        (joint_dict["left_hip_yaw"], joint_dict["left_hip_roll"]),
        (joint_dict["left_hip_yaw"], joint_dict["left_hip_pitch"]),
        (joint_dict["left_hip_yaw"], joint_dict["left_knee"]),
        (joint_dict["left_hip_yaw"], joint_dict["torso"]),
        # left hip roll
        (joint_dict["left_hip_roll"], joint_dict["left_hip_yaw"]),
        (joint_dict["left_hip_roll"], joint_dict["left_hip_pitch"]),
        (joint_dict["left_hip_roll"], joint_dict["left_knee"]),
        (joint_dict["left_hip_roll"], joint_dict["torso"]),
        # left hip pitch
        (joint_dict["left_hip_pitch"], joint_dict["left_hip_yaw"]),
        (joint_dict["left_hip_pitch"], joint_dict["left_hip_roll"]),
        (joint_dict["left_hip_pitch"], joint_dict["left_knee"]),
        (joint_dict["left_hip_pitch"], joint_dict["torso"]),
        # left knee
        (joint_dict["left_knee"], joint_dict["left_ankle"]),
        (joint_dict["left_knee"], joint_dict["torso"]),
        (joint_dict["left_knee"], joint_dict["left_hip_yaw"]),
        (joint_dict["left_knee"], joint_dict["left_hip_roll"]),
        (joint_dict["left_knee"], joint_dict["left_hip_pitch"]),
        # left ankle
        (joint_dict["left_ankle"], joint_dict["left_knee"]),
        # right hip yaw
        (joint_dict["right_hip_yaw"], joint_dict["right_hip_roll"]),
        (joint_dict["right_hip_yaw"], joint_dict["right_hip_pitch"]),
        (joint_dict["right_hip_yaw"], joint_dict["right_knee"]),
        (joint_dict["right_hip_yaw"], joint_dict["torso"]),
        # right hip roll
        (joint_dict["right_hip_roll"], joint_dict["right_hip_yaw"]),
        (joint_dict["right_hip_roll"], joint_dict["right_hip_pitch"]),
        (joint_dict["right_hip_roll"], joint_dict["right_knee"]),
        (joint_dict["right_hip_roll"], joint_dict["torso"]),
        # right hip pitch
        (joint_dict["right_hip_pitch"], joint_dict["right_hip_yaw"]),
        (joint_dict["right_hip_pitch"], joint_dict["right_hip_roll"]),
        (joint_dict["right_hip_pitch"], joint_dict["right_knee"]),
        (joint_dict["right_hip_pitch"], joint_dict["torso"]),
        # right knee
        (joint_dict["right_knee"], joint_dict["right_ankle"]),
        (joint_dict["right_knee"], joint_dict["torso"]),
        (joint_dict["right_knee"], joint_dict["right_hip_yaw"]),
        (joint_dict["right_knee"], joint_dict["right_hip_roll"]),
        (joint_dict["right_knee"], joint_dict["right_hip_pitch"]),
        # right ankle
        (joint_dict["right_ankle"], joint_dict["right_knee"]),
        # torso
        (joint_dict["torso"], joint_dict["left_hip_yaw"]),
        (joint_dict["torso"], joint_dict["right_hip_yaw"]),
        (joint_dict["torso"], joint_dict["left_hip_roll"]),
        (joint_dict["torso"], joint_dict["right_hip_roll"]),
        (joint_dict["torso"], joint_dict["left_hip_pitch"]),
        (joint_dict["torso"], joint_dict["right_hip_pitch"]),
        (joint_dict["torso"], joint_dict["left_shoulder_pitch"]),
        (joint_dict["torso"], joint_dict["right_shoulder_pitch"]),
        (joint_dict["torso"], joint_dict["left_shoulder_roll"]),
        (joint_dict["torso"], joint_dict["right_shoulder_roll"]),
        (joint_dict["torso"], joint_dict["left_shoulder_yaw"]),
        (joint_dict["torso"], joint_dict["right_shoulder_yaw"]),
        # left shoulder pitch
        (joint_dict["left_shoulder_pitch"], joint_dict["torso"]),
        (joint_dict["left_shoulder_pitch"], joint_dict["left_shoulder_roll"]),
        (joint_dict["left_shoulder_pitch"], joint_dict["left_shoulder_yaw"]),
        (joint_dict["left_shoulder_pitch"], joint_dict["left_elbow"]),
        # left shoulder roll
        (joint_dict["left_shoulder_roll"], joint_dict["torso"]),
        (joint_dict["left_shoulder_roll"], joint_dict["left_shoulder_pitch"]),
        (joint_dict["left_shoulder_roll"], joint_dict["left_shoulder_yaw"]),
        (joint_dict["left_shoulder_roll"], joint_dict["left_elbow"]),
        # left shoulder yaw
        (joint_dict["left_shoulder_yaw"], joint_dict["torso"]),
        (joint_dict["left_shoulder_yaw"], joint_dict["left_shoulder_roll"]),
        (joint_dict["left_shoulder_yaw"], joint_dict["left_shoulder_pitch"]),
        (joint_dict["left_shoulder_yaw"], joint_dict["left_elbow"]),
        # left elbow
        (joint_dict["left_elbow"], joint_dict["left_shoulder_roll"]),
        (joint_dict["left_elbow"], joint_dict["left_shoulder_pitch"]),
        (joint_dict["left_elbow"], joint_dict["left_shoulder_yaw"]),
        # right shoulder pitch
        (joint_dict["right_shoulder_pitch"], joint_dict["torso"]),
        (joint_dict["right_shoulder_pitch"], joint_dict["right_shoulder_roll"]),
        (joint_dict["right_shoulder_pitch"], joint_dict["right_shoulder_yaw"]),
        (joint_dict["right_shoulder_pitch"], joint_dict["right_elbow"]),
        # right shoulder roll
        (joint_dict["right_shoulder_roll"], joint_dict["torso"]),
        (joint_dict["right_shoulder_roll"], joint_dict["right_shoulder_pitch"]),
        (joint_dict["right_shoulder_roll"], joint_dict["right_shoulder_yaw"]),
        (joint_dict["right_shoulder_roll"], joint_dict["right_elbow"]),
        # right shoulder yaw
        (joint_dict["right_shoulder_yaw"], joint_dict["torso"]),
        (joint_dict["right_shoulder_yaw"], joint_dict["right_shoulder_roll"]),
        (joint_dict["right_shoulder_yaw"], joint_dict["right_shoulder_pitch"]),
        (joint_dict["right_shoulder_yaw"], joint_dict["right_elbow"]),
        # right elbow
        (joint_dict["right_elbow"], joint_dict["right_shoulder_roll"]),
        (joint_dict["right_elbow"], joint_dict["right_shoulder_pitch"]),
    ]

    num_nodes = len(joint_dict)
    num_edges = len(edge_list)

    # Define body tree structure and joint order - using torch tensors
    # Only include main bodies for simplicity; you can extend to every link
    body_tree = {
        "pelvis": {
            "pos": torch.tensor([0.0, 0.0, 0.0], device=device), "quat": None, "joint": "free_base", "children": [
                {"name": "left_hip_yaw_link", "pos": torch.tensor([0.0, 0.0875, -0.1742], device=device), "joint": "left_hip_yaw", "children": [
                    {"name": "left_hip_roll_link", "pos": torch.tensor([0.039468, 0.0, 0.0], device=device), "joint": "left_hip_roll", "children": [
                        {"name": "left_hip_pitch_link", "pos": torch.tensor([0.0, 0.11536, 0.0], device=device), "joint": "left_hip_pitch", "children": [
                            {"name": "left_knee_link", "pos": torch.tensor([0.0, 0.0, -0.4], device=device), "joint": "left_knee", "children": [
                                {"name": "left_ankle_link", "pos": torch.tensor([0.0, 0.0, -0.4], device=device), "joint": "left_ankle", "children": []}
                            ]}
                        ]}
                    ]}
                ]},
                {"name": "right_hip_yaw_link", "pos": torch.tensor([0.0, -0.0875, -0.1742], device=device), "joint": "right_hip_yaw", "children": [
                    {"name": "right_hip_roll_link", "pos": torch.tensor([0.039468, 0.0, 0.0], device=device), "joint": "right_hip_roll", "children": [
                        {"name": "right_hip_pitch_link", "pos": torch.tensor([0.0, -0.11536, 0.0], device=device), "joint": "right_hip_pitch", "children": [
                            {"name": "right_knee_link", "pos": torch.tensor([0.0, 0.0, -0.4], device=device), "joint": "right_knee", "children": [
                                {"name": "right_ankle_link", "pos": torch.tensor([0.0, 0.0, -0.4], device=device), "joint": "right_ankle", "children": []}
                            ]}
                        ]}
                    ]}
                ]},
                {"name": "torso_link", "pos": torch.tensor([0.0, 0.0, 0.0], device=device), "joint": "torso", "children": [
                    {"name": "left_shoulder_pitch_link", "pos": torch.tensor([0.0055, 0.15535, 0.42999], device=device), "joint": "left_shoulder_pitch", "children": [
                        {"name": "left_shoulder_roll_link", "pos": torch.tensor([-0.0055, 0.0565, -0.0165], device=device), "joint": "left_shoulder_roll", "children": [
                            {"name": "left_shoulder_yaw_link", "pos": torch.tensor([0.0, 0.0, -0.1343], device=device), "joint": "left_shoulder_yaw", "children": [
                                {"name": "left_elbow_link", "pos": torch.tensor([0.0185, 0.0, -0.198], device=device), "joint": "left_elbow", "children": []}
                            ]}
                        ]}
                    ]},
                    {"name": "right_shoulder_pitch_link", "pos": torch.tensor([0.0055, -0.15535, 0.42999], device=device), "joint": "right_shoulder_pitch", "children": [
                        {"name": "right_shoulder_roll_link", "pos": torch.tensor([-0.0055, -0.0565, -0.0165], device=device), "joint": "right_shoulder_roll", "children": [
                            {"name": "right_shoulder_yaw_link", "pos": torch.tensor([0.0, 0.0, -0.1343], device=device), "joint": "right_shoulder_yaw", "children": [
                                {"name": "right_elbow_link", "pos": torch.tensor([0.0185, 0.0, -0.198], device=device), "joint": "right_elbow", "children": []}
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

    # Joint axes (from XML) - using torch tensors
    joint_axes = {
            "left_hip_yaw": torch.tensor([0.0, 0.0, 1.0], device=device),
            "left_hip_roll": torch.tensor([1.0, 0.0, 0.0], device=device),
            "left_hip_pitch": torch.tensor([0.0, 1.0, 0.0], device=device),
            "left_knee": torch.tensor([0.0, 1.0, 0.0], device=device),
            "left_ankle": torch.tensor([0.0, 1.0, 0.0], device=device),
            "right_hip_yaw": torch.tensor([0.0, 0.0, 1.0], device=device),
            "right_hip_roll": torch.tensor([1.0, 0.0, 0.0], device=device),
            "right_hip_pitch": torch.tensor([0.0, 1.0, 0.0], device=device),
            "right_knee": torch.tensor([0.0, 1.0, 0.0], device=device),
            "right_ankle": torch.tensor([0.0, 1.0, 0.0], device=device),
            "torso": torch.tensor([0.0, 0.0, 1.0], device=device),
            "left_shoulder_pitch": torch.tensor([0.0, 1.0, 0.0], device=device),
            "left_shoulder_roll": torch.tensor([1.0, 0.0, 0.0], device=device),
            "left_shoulder_yaw": torch.tensor([0.0, 0.0, 1.0], device=device),
            "left_elbow": torch.tensor([0.0, 1.0, 0.0], device=device),
            "right_shoulder_pitch": torch.tensor([0.0, 1.0, 0.0], device=device),
            "right_shoulder_roll": torch.tensor([1.0, 0.0, 0.0], device=device),
            "right_shoulder_yaw": torch.tensor([0.0, 0.0, 1.0], device=device),
            "right_elbow": torch.tensor([0.0, 1.0, 0.0], device=device)
    }

class FastH1FK:
    def __init__(self, body_tree, joint_indices, joint_axes, joint_names_ordered):
        """
        Pre-flatten body_tree into arrays for fast iterative FK.
        """
        self.body_tree = body_tree
        self.joint_indices = joint_indices
        self.joint_axes = joint_axes
        self.joint_names_ordered = joint_names_ordered

        (
            self.joint_names,
            self.parent,
            self.pos_offset,
            self.axis,
            self.joint_type,
        ) = self._flatten_tree(body_tree)

        self.name_to_index = {n: i for i, n in enumerate(self.joint_names)}
        self.output_index = {n: i for i, n in enumerate(joint_names_ordered)}

    def _flatten_tree(self, body_tree):
        joint_names = []
        parent = []
        pos_offset = []
        axis = []
        joint_type = []

        def recurse(node, parent_idx):
            jname = node.get("joint", node.get("name"))
            idx = len(joint_names)

            joint_names.append(jname)
            parent.append(parent_idx)
            pos_offset.append(node.get("pos", torch.zeros(3, device=device)))
            if jname == "free_base":
                joint_type.append("free")
                axis.append(torch.zeros(3, device=device))
            elif jname in self.joint_axes:
                joint_type.append("revolute")
                axis.append(self.joint_axes[jname])
            else:
                joint_type.append("fixed")
                axis.append(torch.zeros(3, device=device))

            for child in node.get("children", []):
                recurse(child, idx)

        root = list(body_tree.values())[0]
        recurse(root, -1)

        parent = torch.tensor(parent, dtype=torch.long)
        pos_offset = torch.stack([
            t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float)
            for t in pos_offset
        ])
        axis = torch.stack(axis)

        return joint_names, parent, pos_offset, axis, joint_type

    def _quat_to_matrix(self, quat, device):
        """
        quat: [B,4] [w,x,y,z]
        return: [B,3,3]
        """
        quat = F.normalize(quat, dim=1)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        B = quat.shape[0]

        R = torch.zeros(B, 3, 3, device=device, dtype=quat.dtype)
        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - z*w)
        R[:, 0, 2] = 2 * (x*z + y*w)
        R[:, 1, 0] = 2 * (x*y + z*w)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - x*w)
        R[:, 2, 0] = 2 * (x*z - y*w)
        R[:, 2, 1] = 2 * (y*z + x*w)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R

    def _axis_angle_to_matrix(self, axis, angle, device):
        """
        axis: [3], angle: [B]
        return: [B,3,3]
        """
        # Ensure axis lives on the same device/dtype as angle to avoid CPU/CUDA mismatch
        if isinstance(axis, torch.Tensor):
            axis = axis.to(device=device, dtype=angle.dtype)
        else:
            axis = torch.as_tensor(axis, device=device, dtype=angle.dtype)
        axis = axis / (axis.norm() + 1e-9)
        x, y, z = axis
        cos, sin = torch.cos(angle), torch.sin(angle)
        B = angle.shape[0]
        R = torch.zeros(B, 3, 3, device=device, dtype=angle.dtype)

        R[:, 0, 0] = cos + x*x*(1-cos)
        R[:, 0, 1] = x*y*(1-cos) - z*sin
        R[:, 0, 2] = x*z*(1-cos) + y*sin
        R[:, 1, 0] = y*x*(1-cos) + z*sin
        R[:, 1, 1] = cos + y*y*(1-cos)
        R[:, 1, 2] = y*z*(1-cos) - x*sin
        R[:, 2, 0] = z*x*(1-cos) - y*sin
        R[:, 2, 1] = z*y*(1-cos) + x*sin
        R[:, 2, 2] = cos + z*z*(1-cos)
        return R

    def fk_joint_positions(self, qpos: torch.tensor):
        """
        Compute batched FK joint positions (ordered).
        qpos: [B, qpos_dim]
        return: [B, num_joints, 3] (aligned with joint_names_ordered)
        """
        B = qpos.shape[0]
        J = len(self.joint_names)
        device, dtype = qpos.device, qpos.dtype

        joint_pos = torch.zeros(B, J, 3, device=device, dtype=dtype)
        joint_rot = torch.eye(3, device=device, dtype=dtype).expand(B, 3, 3).unsqueeze(1).repeat(1, J, 1, 1).clone()

        for j, jname in enumerate(self.joint_names):
            pj = self.parent[j]

            if self.joint_type[j] == "free":
                base = qpos[:, self.joint_indices[jname]]  # [B,7]
                pos, quat = base[:, :3], base[:, 3:7]
                R = self._quat_to_matrix(quat, device)
                joint_pos[:, j] = pos
                joint_rot[:, j] = R

            else:
                R_parent = joint_rot[:, pj]
                p_parent = joint_pos[:, pj]
                offset = self.pos_offset[j].to(device, dtype)
                p_local = (R_parent @ offset) + p_parent
                joint_pos[:, j] = p_local
                R_joint = R_parent
                if self.joint_type[j] == "revolute":
                    angle = qpos[:, self.joint_indices[jname]]
                    R_rel = self._axis_angle_to_matrix(self.axis[j], angle, device)
                    R_joint = R_parent @ R_rel
                joint_rot[:, j] = R_joint

        # Reorder into [B, ordered_J, 3]
        num_out = len(self.joint_names_ordered)
        out = torch.zeros(B, num_out, 3, device=device, dtype=dtype)
        for jname, j_out in self.output_index.items():
            if jname in self.name_to_index:
                out[:, j_out] = joint_pos[:, self.name_to_index[jname]]

        return out

    def get_joint_index_mapping(self):
        return self.output_index

h1 = H1()

h1_fk = FastH1FK(
    H1.body_tree,
    H1.joint_indices,
    H1.joint_axes,
    H1.joint_names_ordered,
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx

    # Create reverse mapping from joint numbers to names
    number_to_name = {v: k for k, v in h1.joint_dict.items()}
    
    # assume edge_list is already built
    G = nx.DiGraph()  # directed graph (you can use nx.Graph() if you want undirected)

    G.add_edges_from(h1.edge_list)

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # or nx.kamada_kawai_layout(G) for a nicer look
    
    # Create labels mapping node numbers to joint names
    labels = {node: number_to_name.get(node, str(node)) for node in G.nodes()}
    
    nx.draw(
        G, pos,
        with_labels=True,
        labels=labels,
        node_size=1500,
        node_color="lightblue",
        font_size=8,
        font_weight="bold",
        arrowsize=12
    )
    plt.savefig("h1_graph.png")
