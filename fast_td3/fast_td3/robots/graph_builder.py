import enum

import torch



class NodeType(enum.IntEnum):
    JOINT = 0
    OBJECT = 1


class EdgeType(enum.IntEnum):
    JOINT_TO_JOINT = 0
    JOINT_TO_OBJECT = 1


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


joint_connections = [
    # left hip yaw
    (JOINT.left_hip_yaw, JOINT.torso),
    (JOINT.left_hip_yaw, JOINT.left_hip_roll),
    (JOINT.left_hip_yaw, JOINT.left_hip_pitch),
    (JOINT.left_hip_yaw, JOINT.left_knee),
    (JOINT.left_hip_yaw, JOINT.left_ankle),
    # left hip roll
    (JOINT.left_hip_roll, JOINT.torso),
    (JOINT.left_hip_roll, JOINT.left_hip_yaw),
    (JOINT.left_hip_roll, JOINT.left_hip_pitch),
    (JOINT.left_hip_roll, JOINT.left_knee),
    (JOINT.left_hip_roll, JOINT.left_ankle),
    # left hip pitch
    (JOINT.left_hip_pitch, JOINT.torso),
    (JOINT.left_hip_pitch, JOINT.left_hip_yaw),
    (JOINT.left_hip_pitch, JOINT.left_hip_roll),
    (JOINT.left_hip_pitch, JOINT.left_knee),
    (JOINT.left_hip_pitch, JOINT.left_ankle),
    # left knee
    (JOINT.left_knee, JOINT.torso),
    (JOINT.left_knee, JOINT.left_hip_yaw),
    (JOINT.left_knee, JOINT.left_hip_roll),
    (JOINT.left_knee, JOINT.left_hip_pitch),
    (JOINT.left_knee, JOINT.left_ankle),
    # left ankle
    (JOINT.left_ankle, JOINT.left_knee),
    (JOINT.left_ankle, JOINT.left_hip_yaw),
    (JOINT.left_ankle, JOINT.left_hip_roll),
    (JOINT.left_ankle, JOINT.left_hip_pitch),
    # right hip yaw
    (JOINT.right_hip_yaw, JOINT.torso),
    (JOINT.right_hip_yaw, JOINT.right_hip_roll),
    (JOINT.right_hip_yaw, JOINT.right_hip_pitch),
    (JOINT.right_hip_yaw, JOINT.right_knee),
    (JOINT.right_hip_yaw, JOINT.right_ankle),
    # right hip roll
    (JOINT.right_hip_roll, JOINT.torso),
    (JOINT.right_hip_roll, JOINT.right_hip_yaw),
    (JOINT.right_hip_roll, JOINT.right_hip_pitch),
    (JOINT.right_hip_roll, JOINT.right_knee),
    (JOINT.right_hip_roll, JOINT.right_ankle),
    # right hip pitch
    (JOINT.right_hip_pitch, JOINT.torso),
    (JOINT.right_hip_pitch, JOINT.right_hip_yaw),
    (JOINT.right_hip_pitch, JOINT.right_hip_roll),
    (JOINT.right_hip_pitch, JOINT.right_knee),
    (JOINT.right_hip_pitch, JOINT.right_ankle),
    # right knee
    (JOINT.right_knee, JOINT.torso),
    (JOINT.right_knee, JOINT.right_hip_yaw),
    (JOINT.right_knee, JOINT.right_hip_roll),
    (JOINT.right_knee, JOINT.right_hip_pitch),
    (JOINT.right_knee, JOINT.right_ankle),
    # right ankle
    (JOINT.right_ankle, JOINT.right_knee),
    (JOINT.right_ankle, JOINT.right_hip_yaw),
    (JOINT.right_ankle, JOINT.right_hip_roll),
    (JOINT.right_ankle, JOINT.right_hip_pitch),
    # torso
    (JOINT.torso, JOINT.left_hip_yaw),
    (JOINT.torso, JOINT.right_hip_yaw),
    (JOINT.torso, JOINT.left_hip_roll),
    (JOINT.torso, JOINT.right_hip_roll),
    (JOINT.torso, JOINT.left_hip_pitch),
    (JOINT.torso, JOINT.right_hip_pitch),
    (JOINT.torso, JOINT.left_shoulder_pitch),
    (JOINT.torso, JOINT.right_shoulder_pitch),
    (JOINT.torso, JOINT.left_shoulder_roll),
    (JOINT.torso, JOINT.right_shoulder_roll),
    (JOINT.torso, JOINT.left_shoulder_yaw),
    (JOINT.torso, JOINT.right_shoulder_yaw),
    (JOINT.torso, JOINT.left_elbow),
    (JOINT.torso, JOINT.left_elbow),
    (JOINT.torso, JOINT.left_ankle),
    (JOINT.torso, JOINT.right_ankle),
    # left shoulder pitch
    (JOINT.left_shoulder_pitch, JOINT.torso),
    (JOINT.left_shoulder_pitch, JOINT.left_shoulder_roll),
    (JOINT.left_shoulder_pitch, JOINT.left_shoulder_yaw),
    (JOINT.left_shoulder_pitch, JOINT.left_elbow),
    # left shoulder yaw
    (JOINT.left_shoulder_yaw, JOINT.torso),
    (JOINT.left_shoulder_yaw, JOINT.left_shoulder_roll),
    (JOINT.left_shoulder_yaw, JOINT.left_shoulder_pitch),
    (JOINT.left_shoulder_yaw, JOINT.left_elbow),
    # left shoulder roll
    (JOINT.left_shoulder_roll, JOINT.torso),
    (JOINT.left_shoulder_roll, JOINT.left_shoulder_pitch),
    (JOINT.left_shoulder_roll, JOINT.left_shoulder_yaw),
    (JOINT.left_shoulder_roll, JOINT.left_elbow),
    # left elbow
    (JOINT.left_elbow, JOINT.left_shoulder_roll),
    (JOINT.left_elbow, JOINT.left_shoulder_pitch),
    (JOINT.left_elbow, JOINT.left_shoulder_yaw),
    # right shoulder pitch
    (JOINT.right_shoulder_pitch, JOINT.torso),
    (JOINT.right_shoulder_pitch, JOINT.right_shoulder_roll),
    (JOINT.right_shoulder_pitch, JOINT.right_shoulder_yaw),
    (JOINT.right_shoulder_pitch, JOINT.right_elbow),
    # right shoulder yaw
    (JOINT.right_shoulder_yaw, JOINT.torso),
    (JOINT.right_shoulder_yaw, JOINT.right_shoulder_roll),
    (JOINT.right_shoulder_yaw, JOINT.right_shoulder_pitch),
    (JOINT.right_shoulder_yaw, JOINT.right_elbow),
    # right shoulder roll
    (JOINT.right_shoulder_roll, JOINT.torso),
    (JOINT.right_shoulder_roll, JOINT.right_shoulder_pitch),
    (JOINT.right_shoulder_roll, JOINT.right_shoulder_yaw),
    (JOINT.right_shoulder_roll, JOINT.right_elbow),
    # right elbow
    (JOINT.right_elbow, JOINT.right_shoulder_roll),
    (JOINT.right_elbow, JOINT.right_shoulder_pitch),
    (JOINT.right_elbow, JOINT.right_shoulder_yaw),
]

# TODO: this currently works for h1, need to generalize for other robots
class GraphBuilder:
    def __init__(self):
        pass

    def generate_edge_index(self, batch_size, device='cpu'):
        src, dst = zip(*joint_connections)  # Unpack edge list into two tuples
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)

        print(joint_connections)

        # Create batch offsets and expand edges in one operation
        offsets = torch.arange(batch_size) * len(JOINT)
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)

        return torch.stack([src_batch, dst_batch]), len(JOINT), len(joint_connections)


    def get_joint_name(self, joint_id):
        """Convert joint ID to readable name"""
        joint_names = {
            JOINT.left_hip_yaw: "L_Hip_Yaw",
            JOINT.left_hip_roll: "L_Hip_Roll", 
            JOINT.left_hip_pitch: "L_Hip_Pitch",
            JOINT.left_knee: "L_Knee",
            JOINT.left_ankle: "L_Ankle",
            JOINT.right_hip_yaw: "R_Hip_Yaw",
            JOINT.right_hip_roll: "R_Hip_Roll",
            JOINT.right_hip_pitch: "R_Hip_Pitch", 
            JOINT.right_knee: "R_Knee",
            JOINT.right_ankle: "R_Ankle",
            JOINT.torso: "Torso",
            JOINT.left_shoulder_pitch: "L_Shldr_Pitch",
            JOINT.left_shoulder_roll: "L_Shldr_Roll",
            JOINT.left_shoulder_yaw: "L_Shldr_Yaw",
            JOINT.left_elbow: "L_Elbow",
            JOINT.right_shoulder_pitch: "R_Shldr_Pitch",
            JOINT.right_shoulder_roll: "R_Shldr_Roll", 
            JOINT.right_shoulder_yaw: "R_Shldr_Yaw",
            JOINT.right_elbow: "R_Elbow"
        }
        return joint_names.get(joint_id, f"Joint_{joint_id}")

    def get_robot_layout_positions(self):
        """Define positions to create a robot-like symmetric layout"""
        positions = {}
        
        # Torso at center
        positions[JOINT.torso] = (0, 0)
        
        # Left arm (from torso perspective)
        positions[JOINT.left_shoulder_pitch] = (-1, 0.3)
        positions[JOINT.left_shoulder_roll] = (-1.25, 0.6)
        positions[JOINT.left_shoulder_yaw] = (-1, 0)
        positions[JOINT.left_elbow] = (-1.5, 0.3)
        
        # Right arm (symmetric)
        positions[JOINT.right_shoulder_pitch] = (1, 0.3)
        positions[JOINT.right_shoulder_roll] = (1.25, 0.6)
        positions[JOINT.right_shoulder_yaw] = (1, 0)
        positions[JOINT.right_elbow] = (1.5, 0.3)

        # Left leg
        positions[JOINT.left_hip_yaw] = (-0.5, -0.5)
        positions[JOINT.left_hip_roll] = (-0.7, -1)
        positions[JOINT.left_hip_pitch] = (-0.3, -1)
        positions[JOINT.left_knee] = (-0.5, -2)
        positions[JOINT.left_ankle] = (-0.5, -2.5)
        
        # Right leg (symmetric)
        positions[JOINT.right_hip_yaw] = (0.5, -0.5)
        positions[JOINT.right_hip_roll] = (0.7, -1)
        positions[JOINT.right_hip_pitch] = (0.3, -1)
        positions[JOINT.right_knee] = (0.5, -2)
        positions[JOINT.right_ankle] = (0.5, -2.5)

        return positions

    def get_connection_type(self, joint1_id, joint2_id):
        """Classify the type of connection between two joints"""
        joint_names = {
            JOINT.left_hip_yaw: "left_hip_yaw",
            JOINT.left_hip_roll: "left_hip_roll", 
            JOINT.left_hip_pitch: "left_hip_pitch",
            JOINT.left_knee: "left_knee",
            JOINT.left_ankle: "left_ankle",
            JOINT.right_hip_yaw: "right_hip_yaw",
            JOINT.right_hip_roll: "right_hip_roll",
            JOINT.right_hip_pitch: "right_hip_pitch", 
            JOINT.right_knee: "right_knee",
            JOINT.right_ankle: "right_ankle",
            JOINT.torso: "torso",
            JOINT.left_shoulder_pitch: "left_shoulder_pitch",
            JOINT.left_shoulder_roll: "left_shoulder_roll",
            JOINT.left_shoulder_yaw: "left_shoulder_yaw",
            JOINT.left_elbow: "left_elbow",
            JOINT.right_shoulder_pitch: "right_shoulder_pitch",
            JOINT.right_shoulder_roll: "right_shoulder_roll", 
            JOINT.right_shoulder_yaw: "right_shoulder_yaw",
            JOINT.right_elbow: "right_elbow"
        }
        
        name1 = joint_names.get(joint1_id, "unknown")
        name2 = joint_names.get(joint2_id, "unknown")
        
        # Torso connections (highest priority)
        if "torso" in [name1, name2]:
            other_joint = name1 if "torso" not in name1 else name2
            if "ankle" in other_joint:
                return "torso_ankle"
            elif "hip" in other_joint:
                return "torso_hip"
            elif "shoulder" in other_joint:
                return "torso_shoulder"
            elif "elbow" in other_joint:
                return "torso_elbow"
            else:
                return "torso_other"
        
        # Left leg connections
        elif "left" in name1 and "left" in name2:
            if ("hip" in name1 and "hip" in name2):
                return "left_hip_internal"
            elif ("hip" in name1 and "knee" in name2) or ("knee" in name1 and "hip" in name2):
                return "left_hip_knee"
            elif ("knee" in name1 and "ankle" in name2) or ("ankle" in name1 and "knee" in name2):
                return "left_knee_ankle"
            else:
                return "left_leg_other"
        
        # Right leg connections
        elif "right" in name1 and "right" in name2:
            if ("hip" in name1 and "hip" in name2):
                return "right_hip_internal"
            elif ("hip" in name1 and "knee" in name2) or ("knee" in name1 and "hip" in name2):
                return "right_hip_knee"
            elif ("knee" in name1 and "ankle" in name2) or ("ankle" in name1 and "knee" in name2):
                return "right_knee_ankle"
            else:
                return "right_leg_other"
        
        # Left arm connections
        elif "left" in name1 and "left" in name2:
            if "shoulder" in name1 and "shoulder" in name2:
                return "left_shoulder_internal"
            elif ("shoulder" in name1 and "elbow" in name2) or ("elbow" in name1 and "shoulder" in name2):
                return "left_shoulder_elbow"
            else:
                return "left_arm_other"
        
        # Right arm connections
        elif "right" in name1 and "right" in name2:
            if "shoulder" in name1 and "shoulder" in name2:
                return "right_shoulder_internal"
            elif ("shoulder" in name1 and "elbow" in name2) or ("elbow" in name1 and "shoulder" in name2):
                return "right_shoulder_elbow"
            else:
                return "right_arm_other"
        
        # Cross-body connections
        else:
            return "cross_body"

    def visualize_graph(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        G = nx.DiGraph()
        
        # Add nodes with joint names as labels
        for joint_id in range(19):  # 0 to 18
            joint_name = self.get_joint_name(joint_id)
            G.add_node(joint_id, label=joint_name)
        
        # Add edges
        for edge in joint_connections:
            G.add_edge(edge[0], edge[1])

        # Use custom robot-like layout
        pos = self.get_robot_layout_positions()
        
        # Create labels dictionary
        labels = {joint_id: self.get_joint_name(joint_id) for joint_id in G.nodes()}
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Define color scheme for different connection types
        connection_colors = {
            "torso_ankle": "#FF0000",      # Red - Torso to ankle
            "torso_hip": "#FF6600",        # Orange - Torso to hip
            "torso_shoulder": "#0066FF",   # Blue - Torso to shoulder
            "torso_elbow": "#0033CC",      # Dark blue - Torso to elbow
            "torso_other": "#666666",      # Gray - Other torso connections
            
            "left_hip_internal": "#00CC00",    # Green - Left hip internal
            "left_hip_knee": "#00AA00",        # Dark green - Left hip to knee
            "left_knee_ankle": "#008800",      # Darker green - Left knee to ankle
            "left_leg_other": "#004400",       # Very dark green - Other left leg
            
            "right_hip_internal": "#FFCC00",   # Yellow - Right hip internal
            "right_hip_knee": "#FFAA00",       # Orange-yellow - Right hip to knee
            "right_knee_ankle": "#FF8800",     # Orange - Right knee to ankle
            "right_leg_other": "#CC6600",      # Dark orange - Other right leg
            
            "left_shoulder_internal": "#00CCCC",   # Cyan - Left shoulder internal
            "left_shoulder_elbow": "#0099AA",      # Teal - Left shoulder to elbow
            "left_arm_other": "#006666",           # Dark teal - Other left arm
            
            "right_shoulder_internal": "#CC00CC",  # Magenta - Right shoulder internal
            "right_shoulder_elbow": "#AA0099",     # Purple - Right shoulder to elbow
            "right_arm_other": "#660066",          # Dark purple - Other right arm
            
            "cross_body": "#333333"                # Dark gray - Cross body connections
        }
        
        # Categorize edges by connection type
        edge_groups = {}
        for edge in G.edges():
            conn_type = self.get_connection_type(edge[0], edge[1])
            if conn_type not in edge_groups:
                edge_groups[conn_type] = []
            edge_groups[conn_type].append(edge)
        
        # Draw nodes with better styling and color coding
        node_colors = []
        for joint_id in range(19):
            joint_name = self.get_joint_name(joint_id)
            if "Torso" in joint_name:
                node_colors.append("#FFD700")  # Gold for torso
            elif "L_" in joint_name and ("Hip" in joint_name or "Knee" in joint_name or "Ankle" in joint_name):
                node_colors.append("#90EE90")  # Light green for left leg
            elif "R_" in joint_name and ("Hip" in joint_name or "Knee" in joint_name or "Ankle" in joint_name):
                node_colors.append("#FFE4B5")  # Light orange for right leg
            elif "L_" in joint_name and ("Shldr" in joint_name or "Elbow" in joint_name):
                node_colors.append("#87CEEB")  # Sky blue for left arm
            elif "R_" in joint_name and ("Shldr" in joint_name or "Elbow" in joint_name):
                node_colors.append("#DDA0DD")  # Plum for right arm
            else:
                node_colors.append("#D3D3D3")  # Light gray for others
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, 
                              alpha=0.9, linewidths=2, edgecolors='black')
        
        # Draw edges by type with different colors and styles
        for conn_type, edges in edge_groups.items():
            color = connection_colors.get(conn_type, "#999999")
            
            # Different line styles for different connection types
            if "torso" in conn_type:
                style = "solid"
                width = 2.5
                alpha = 0.8
            elif "internal" in conn_type:
                style = "dashed"
                width = 1.5
                alpha = 0.7
            else:
                style = "solid"
                width = 1.8
                alpha = 0.7
            
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, 
                                  arrows=True, arrowsize=12, alpha=alpha, 
                                  width=width, style=style, arrowstyle='->')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", 
                               font_color='black')
        
        # Create legend
        legend_elements = [
            patches.Patch(color='#FF0000', label='Torso ↔ Ankle'),
            patches.Patch(color='#FF6600', label='Torso ↔ Hip'),
            patches.Patch(color='#0066FF', label='Torso ↔ Shoulder'),
            patches.Patch(color='#00CC00', label='Left Leg Internal'),
            patches.Patch(color='#FFCC00', label='Right Leg Internal'),
            patches.Patch(color='#00CCCC', label='Left Arm Internal'),
            patches.Patch(color='#CC00CC', label='Right Arm Internal'),
            patches.Patch(color='#333333', label='Cross-body'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), 
                 fontsize=10, title="Connection Types", title_fontsize=12)
        
        plt.title("Humanoid Robot Joint Connection Graph\n(Color-coded by Connection Type)", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("robot_graph.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Print connection statistics
        print(f"\nGraph saved as 'robot_graph.png' with {len(G.nodes())} joints and {len(G.edges())} connections")
        print("\nConnection Type Statistics:")
        for conn_type, edges in sorted(edge_groups.items()):
            print(f"  {conn_type}: {len(edges)} connections")
        
        return G

if __name__ == "__main__":
    gb = GraphBuilder()
    # gb.visualize_graph()
    print(gb.generate_edge_index(1))