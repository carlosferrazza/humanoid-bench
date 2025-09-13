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


# TODO: this currently works for h1, need to generalize for other robots
class GraphBuilder:
    def __init__(self, robot="h1"):
        if robot.lower() == "h1":
            self.robot = H1()

    def generate_edge_index(self, batch_size, device="cpu"):
        src, dst = zip(
            *self.robot.joint_connections
        )  # Unpack edge list into two tuples
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)

        # Create batch offsets and expand edges in one operation
        offsets = torch.arange(batch_size) * len(self.robot.JOINT)
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)

        return (
            torch.stack([src_batch, dst_batch]),
            len(self.robot.JOINT),
            len(self.robot.joint_connections),
        )

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
