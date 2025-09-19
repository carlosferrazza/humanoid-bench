import enum
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from .H1 import H1


class NodeType(enum.IntEnum):
    JOINT = 0
    OBJECT = 1


class EdgeType(enum.IntEnum):
    JOINT_TO_JOINT = 0
    JOINT_TO_OBJECT = 1


env_with_object = [
    "h1-push-v0",
    "h1-basketball-v0",
    "h1-package-v0",
    "h1-sit_hard-v0",
    "h1-balance_simple-v0",
]

env_without_object = [
    "h1-walk-v0",
    "h1-reach-v0",
    "h1-hurdle-v0",
    "h1-crawl-v0",
    "h1-maze-v0",
    "h1-highbar_simple-v0",
    "h1-stand-v0",
    "h1-run-v0",
    "h1-sit_simple-v0",
    "h1-stairs-v0",
    "h1-slide-v0",
    "h1-pole-v0",
]


# TODO: this currently works for h1, need to generalize for other robots
class GraphBuilder:
    def __init__(self, env_name, batch_size, device, robot="h1"):
        if robot.lower() == "h1":
            self.robot = H1()
        else:
            raise NotImplementedError(f"Robot {robot} not implemented.")

        self.device = device
        self.env_name = env_name
        self.batch_size = batch_size
        self.edge_index, self.edge_attr, self.node_attr = self._generate_index(batch_size, device)
        self._edge_cache = {}
        self.num_edges = self.robot.joint_connections.__len__()

    def _generate_index(self, batch_size: int, device="cuda"):
        src, dst = zip(*self.robot.joint_connections)

        edge_attr = []

        if self.env_name in env_with_object:
            object_node_id = len(self.robot.JOINT)
            for joint_id in range(len(self.robot.JOINT)):
                src += (joint_id,)
                dst += (object_node_id,)
            # sort src and dst based on src first, then dst
            sorted_edges = sorted(zip(src, dst), key=lambda x: (x[0], x[1]))
            src, dst = zip(*sorted_edges)

            # Create edge_attr: 1 if edge involves object_node_id, else 0
            for s, d in zip(src, dst):
                if s == object_node_id or d == object_node_id:
                    edge_attr.append(1)
                else:
                    edge_attr.append(0)
        else:
            # No object node, all edge_attr are 0
            for s, d in zip(src, dst):
                edge_attr.append(0)

        # Unpack edge list into two tuples
        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=device)
        node_attr = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], dtype=torch.float, device=device)


        # Create batch offsets and expand edges in one operation
        offsets = torch.arange(batch_size, device=device) * len(self.robot.JOINT)
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)
        edge_attr_batch = edge_attr.repeat(batch_size).to(device).unsqueeze(-1)
        node_attr_batch = node_attr.repeat(batch_size).to(device).unsqueeze(-1)

        return torch.stack([src_batch, dst_batch]), edge_attr_batch, node_attr_batch

    # obs = qpos + qvel
    # structure of obs: /home/duckoid/Downloads/humanoid-bench/fast_td3/src/humanoid-bench/humanoid_bench/tasks.py
    def generate_input(self, obs: torch.tensor, xpos: torch.tensor):
        current_batch_size = obs.shape[0]
        if current_batch_size == self.batch_size:
            edge_index, edge_attr, node_attr = self.edge_index, self.edge_attr, self.node_attr
        else:
            assert (
                current_batch_size <= self.batch_size
            ), "Batch size exceeds the maximum batch size."
            if current_batch_size in self._edge_cache:
                edge_index, edge_attr, node_attr = self._edge_cache[current_batch_size]
            else:
                edge_index, edge_attr, node_attr = self._generate_index(current_batch_size, self.device)
                #print(f"edge_index shape: {edge_index.shape}, edge_attr shape: {edge_attr.shape}, node_attr shape: {node_attr.shape}")
                self._edge_cache[current_batch_size] = (edge_index, edge_attr, node_attr)

        if self.env_name in env_with_object:
            assert obs.shape[1] == 63, f"obs shape: {obs.shape}"
            assert xpos.shape[1] == 21, f"xpos shape: {xpos.shape}"
            h = torch.cat(
                [
                    obs[:, 7:27].reshape(-1, 1),
                    obs[:, 39:59].reshape(-1, 1),
                ],
                dim=1,
            )
            x = (xpos[:, 1:] - xpos[:, [0]]).reshape(-1, 3)

            return h, x, edge_index, edge_attr, node_attr
        else:
            assert obs.shape[1] == 51, f"obs shape: {obs.shape}"
            assert xpos.shape[1] == 20, f"xpos shape: {xpos.shape}"
            h = torch.cat(
                [obs[:, 7:26].reshape(-1, 1), obs[:, 32:].reshape(-1, 1)], dim=1
            )
            x = (xpos[:, 1:] - xpos[:, [0]]).reshape(-1, 3)

            return h, x, edge_index, None, None


    def visualize_graph(self):
        G = nx.DiGraph()

        # Add nodes with joint names as labels
        for joint_id in range(19):  # 0 to 18
            joint_name = self.robot.get_joint_name(joint_id)
            G.add_node(joint_id, label=joint_name)

        # Add edges
        for edge in self.robot.joint_connections:
            G.add_edge(edge[0], edge[1])

        # Use custom robot-like layout
        pos = self.robot.get_robot_layout_positions()

        # Create labels dictionary
        labels = {
            joint_id: self.robot.get_joint_name(joint_id) for joint_id in G.nodes()
        }

        # Define color scheme for different connection types
        connection_colors = self.robot.connection_colors

        # Categorize edges by connection type
        edge_groups = {}
        for edge in G.edges():
            conn_type = self.robot.get_connection_type(edge[0], edge[1])
            if conn_type not in edge_groups:
                edge_groups[conn_type] = []
            edge_groups[conn_type].append(edge)

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=400,
            alpha=0.9,
            linewidths=1,
            edgecolors="black",
        )

        # Draw edges by type with different colors and styles
        for conn_type, edges in edge_groups.items():
            color = connection_colors.get(conn_type, "#999999")

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=color,
                arrows=True,
                arrowsize=8,
                alpha=0.8,
                width=1,
                style="solid",
                arrowstyle="->",
            )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos, labels, font_size=4, font_weight="bold", font_color="black"
        )

        plt.title(
            "Humanoid Robot Joint Connection Graph\n(Color-coded by Connection Type)",
            fontsize=10,
            fontweight="bold",
            pad=20,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("robot_graph.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()


if __name__ == "__main__":
    gb = GraphBuilder()
    gb.visualize_graph()
    # print(gb.generate_edge_index(env_name="h1-run-v0", batch_size=4))  # Example usage
