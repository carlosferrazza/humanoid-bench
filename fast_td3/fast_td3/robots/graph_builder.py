import enum
import networkx as nx
import matplotlib.pyplot as plt
import torch

from .h1 import H1


class NodeType(enum.IntEnum):
    JOINT = 0
    OBJECT = 1


class EdgeType(enum.IntEnum):
    JOINT_TO_JOINT = 0
    JOINT_TO_OBJECT = 1


env_with_object = [
    "h1-push-v0",  # medium
    "h1-basketball-v0",  # very hard
    "h1-package-v0",  # medium
    "h1-sit_hard-v0",  # hard
    "h1-balance_simple-v0",  # hard
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
    """Utility to build graph tensors and visualize robot topology.

    Supports optional inclusion of a free object node controlled via:
      - env_name membership in env_with_object (default inference)
      - explicit with_object flag passed to __init__ or visualize_graph()
    """

    def __init__(
        self, env_name, batch_size, device, robot="h1", with_object: bool | None = None
    ):
        if robot.lower() == "h1":
            # Decide whether to construct robot with object: explicit flag overrides env inference
            if with_object is None:
                inferred = env_name in env_with_object
                self.robot = H1(with_object=inferred)
            else:
                self.robot = H1(with_object=with_object)
        else:
            raise NotImplementedError(f"Robot {robot} not implemented.")

        self.device = device
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_edges = self.robot.joint_connections.__len__()

    # obs = qpos + qvel
    # structure of obs: /home/duckoid/Downloads/humanoid-bench/fast_td3/src/humanoid-bench/humanoid_bench/tasks.py
    # @torch.compile(dynamic=True)  # Disabled: not supported on Python 3.12+
    def generate_input(self, obs: torch.tensor, xanchor: torch.tensor):
        """Generate input with root information as global context."""
        assert obs.shape[1] == 51, f"obs shape: {obs.shape}"
        assert xanchor.shape[1] == 20, f"xanchor shape: {xanchor.shape}"

        # Extract root features (13 values)
        root_features = torch.cat(
            [obs[:, 0:7], obs[:, 26:32]],  # root pos (3) + quat (4) = 7  # root vel (6)
            dim=1,
        )  # [batch, 13]

        # Extract joint features
        joint_pos = obs[:, 7:26].reshape(-1, 1)  # [batch*19, 1]
        joint_vel = obs[:, 32:].reshape(-1, 1)  # [batch*19, 1]

        # Concatenate: each joint gets [pos, vel, root_context]
        h = torch.cat(
            [joint_pos, joint_vel],  # 1 value  # 1 value  # 13 values
            dim=1,
        )  # [batch*19, 2]

        # Positions remain relative to root (unchanged)
        x = (xanchor[:, 1:] - xanchor[:, [0]]).reshape(-1, 3)  # [batch*19, 3]

        return h, x, root_features

    # h = qpos concat qvel
    # @torch.compile(dynamic=True)
    def generate_input_for_mixed_type(self, obs: torch.tensor, xanchor: torch.tensor):
        if self.env_name in env_with_object:
            assert xanchor.shape[1] == 21, f"xanchor shape: {xanchor.shape}"
            x_joint = (xanchor[:, 1:20] - xanchor[:, [0]]).reshape(-1, 3)
            x_object = (xanchor[:, 20:] - xanchor[:, [0]]).reshape(-1, 3)

            # # square distance from each joint to object
            # distant_joint_to_object = (
            #     (xanchor[:, 1:20] - xanchor[:, [20]])
            #     .reshape(-1, 3)
            #     .pow(2)
            #     .sum(dim=-1, keepdim=True)
            # )

            h_node = torch.cat(
                [
                    obs[:, 7:26].reshape(-1, 1),
                    obs[:, 39:58].reshape(-1, 1),
                ],
                dim=1,
            )

            # position, quarternion, linear velocity, angular velocity of pelvis and object
            h_object = torch.cat([obs[:, 0:7], obs[:, 26:39], obs[:, 58:64]], dim=1)

            return h_node, h_object, x_joint, x_object

    def visualize_graph(
        self, with_object: bool | None = None, save_path: str = "robot_graph.png"
    ):
        """Visualize the current graph.

        Args:
            with_object: Optional override to include the object node even if env doesn't; if None uses robot.with_object
            save_path: Path to save the generated image.
        """
        if with_object is not None:
            self.robot.set_with_object(with_object)

        G = nx.DiGraph()

        # Determine node ids to add
        num_joint_nodes = len(self.robot.JOINT)
        joint_ids = list(range(num_joint_nodes))
        if self.robot.with_object:
            all_node_ids = joint_ids + [self.robot.OBJECT.free_object]
        else:
            all_node_ids = joint_ids

        # Add nodes with joint/object names as labels
        for nid in all_node_ids:
            G.add_node(nid, label=self.robot.get_joint_name(nid))

        # Add edges
        for edge in self.robot.active_connections:
            G.add_edge(edge[0], edge[1])

        # Use custom robot-like layout (pass override flag)
        pos = self.robot.get_robot_layout_positions(with_object=self.robot.with_object)

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

        title_suffix = " + Object" if self.robot.with_object else ""
        plt.title(
            f"Humanoid Robot Joint Connection Graph{title_suffix}\n(Color-coded by Connection Type)",
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
    gb = GraphBuilder(env_name="h1-run-v0", batch_size=1, device="cpu")
    gb.visualize_graph(save_path="robot_graph_no_object.png")
    gb.visualize_graph(with_object=True, save_path="robot_graph_with_object.png")
    # print(gb.generate_edge_index(env_name="h1-run-v0", batch_size=4))  # Example usage
