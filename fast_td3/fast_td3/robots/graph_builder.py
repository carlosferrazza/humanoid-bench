import enum
import networkx as nx
import matplotlib.pyplot as plt
import torch

from fast_td3.robots.h1 import H1
from fast_td3.robots.g1 import G1


# TODO: this currently works for h1, need to generalize for other robots
class GraphBuilder:
    """Utility to build graph tensors and visualize robot topology."""

    def __init__(
        self, env_name, batch_size, device, robot="h1"
    ):
        robot_lower = robot.lower()
        if robot_lower == "h1":
            self.robot = H1()
        elif robot_lower == "g1":
            self.robot = G1()
        else:
            raise NotImplementedError(f"Robot {robot} not implemented.")

        self.device = device
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_edges = self.robot.joint_connections.__len__()

    @torch.compile(dynamic=True)
    def generate_input(self, obs: torch.tensor, xanchor: torch.tensor):
        """Generate input with root information as global context."""
        assert obs.shape[1] == 51, f"obs shape: {obs.shape}"
        assert xanchor.shape[1] == 20, f"xanchor shape: {xanchor.shape}"

        # Extract joint features
        joint_pos = obs[:, 7:26].reshape(-1, 1)  # [batch*19, 1]
        joint_vel = obs[:, 32:].reshape(-1, 1)  # [batch*19, 1]
        h = torch.cat([joint_vel, joint_pos], dim=1)
        x = (xanchor[:, 1:] - xanchor[:, [0]]).reshape(-1, 3)  # [batch*19, 3]

        # Extract root/object features
        h_object = obs[:, 26:32]
        x_object = xanchor[:, [0]].reshape(-1, 3)  # [batch, 3]yY

        return h, x, h_object, x_object



    def visualize_graph(
        self, save_path: str = "robot_graph.png"
    ):
        """Visualize the current graph.

        Args:
            save_path: Path to save the generated image.
        """
        G = nx.DiGraph()

        # Determine node ids to add
        num_joint_nodes = len(self.robot.JOINT)
        all_node_ids = list(range(num_joint_nodes))

        # Add nodes with joint/object names as labels
        for nid in all_node_ids:
            G.add_node(nid, label=self.robot.get_joint_name(nid))

        # Add edges
        for edge in self.robot.joint_connections:
            G.add_edge(edge[0], edge[1])

        # Use custom robot-like layout
        pos = self.robot.get_robot_layout_positions()

        # Create labels dictionary
        labels = {nid: self.robot.get_joint_name(nid) for nid in G.nodes()}

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
            f"Humanoid Robot Joint Connection Graph\n(Color-coded by Connection Type)",
            fontsize=10,
            fontweight="bold",
            pad=20,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()


if __name__ == "__main__":
    # Example standalone usage for quick visual checks
    print("Visualizing H1 robot...")
    gb_h1 = GraphBuilder(env_name="h1-run-v0", batch_size=1, device="cpu", robot="h1")
    gb_h1.visualize_graph(save_path="h1_robot_graph.png")
    
    print("Visualizing G1 robot...")
    gb_g1 = GraphBuilder(env_name="g1-run-v0", batch_size=1, device="cpu", robot="g1")
    gb_g1.visualize_graph(save_path="g1_robot_graph.png")
