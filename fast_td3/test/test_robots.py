"""Unit tests for robot models and graph builder."""
import unittest
from fast_td3.robots.h1 import H1
from fast_td3.robots.g1 import G1


class TestRobotModels(unittest.TestCase):
    """Test robot model definitions and connections."""

    def test_h1_num_joints(self):
        """Test that H1 has the correct number of joints."""
        robot = H1()
        # H1 has 19 joints (0-18)
        expected_num_joints = 19
        self.assertEqual(robot.num_joints, expected_num_joints, 
                        f"H1 should have {expected_num_joints} joints")

    def test_g1_num_joints(self):
        """Test that G1 has the correct number of joints."""
        robot = G1()
        # G1 has 37 joints (0-36)
        expected_num_joints = 37
        self.assertEqual(robot.num_joints, expected_num_joints, 
                        f"G1 should have {expected_num_joints} joints")

    def test_h1_connections_bidirectional(self):
        """Test that all H1 connections are bidirectional."""
        robot = H1()
        connections = robot.joint_connections
        
        # Check that connections come in pairs (bidirectional)
        for joint1, joint2 in connections:
            reverse_connection = (joint2, joint1)
            self.assertIn(reverse_connection, connections,
                         f"Connection ({joint1}, {joint2}) is not bidirectional")

    def test_g1_connections_bidirectional(self):
        """Test that all G1 connections are bidirectional."""
        robot = G1()
        connections = robot.joint_connections
        
        # Check that connections come in pairs (bidirectional)
        for joint1, joint2 in connections:
            reverse_connection = (joint2, joint1)
            self.assertIn(reverse_connection, connections,
                         f"Connection ({joint1}, {joint2}) is not bidirectional")

    def test_h1_num_edges(self):
        """Test that H1 has edges (connections)."""
        robot = H1()
        num_edges = robot.num_edges
        self.assertGreater(num_edges, 0, "H1 should have at least one edge")
        # Edges should be even (bidirectional)
        self.assertEqual(num_edges % 2, 0, "Number of edges should be even (bidirectional)")

    def test_g1_num_edges(self):
        """Test that G1 has edges (connections)."""
        robot = G1()
        num_edges = robot.num_edges
        self.assertGreater(num_edges, 0, "G1 should have at least one edge")
        # Edges should be even (bidirectional)
        self.assertEqual(num_edges % 2, 0, "Number of edges should be even (bidirectional)")

    def test_h1_joint_indices_valid(self):
        """Test that all H1 joint indices are within valid range."""
        robot = H1()
        connections = robot.joint_connections
        
        for joint1, joint2 in connections:
            self.assertGreaterEqual(joint1, 0, f"Joint index {joint1} is invalid")
            self.assertLess(joint1, robot.num_joints, 
                          f"Joint index {joint1} exceeds num_joints")
            self.assertGreaterEqual(joint2, 0, f"Joint index {joint2} is invalid")
            self.assertLess(joint2, robot.num_joints, 
                          f"Joint index {joint2} exceeds num_joints")

    def test_g1_joint_indices_valid(self):
        """Test that all G1 joint indices are within valid range."""
        robot = G1()
        connections = robot.joint_connections
        
        for joint1, joint2 in connections:
            self.assertGreaterEqual(joint1, 0, f"Joint index {joint1} is invalid")
            self.assertLess(joint1, robot.num_joints, 
                          f"Joint index {joint1} exceeds num_joints")
            self.assertGreaterEqual(joint2, 0, f"Joint index {joint2} is invalid")
            self.assertLess(joint2, robot.num_joints, 
                          f"Joint index {joint2} exceeds num_joints")

    def test_h1_no_self_connections(self):
        """Test that H1 has no self-connections."""
        robot = H1()
        connections = robot.joint_connections
        
        for joint1, joint2 in connections:
            self.assertNotEqual(joint1, joint2, 
                              f"Self-connection found: ({joint1}, {joint2})")

    def test_g1_no_self_connections(self):
        """Test that G1 has no self-connections."""
        robot = G1()
        connections = robot.joint_connections
        
        for joint1, joint2 in connections:
            self.assertNotEqual(joint1, joint2, 
                              f"Self-connection found: ({joint1}, {joint2})")


class TestGraphBuilder(unittest.TestCase):
    """Test graph builder functionality."""

    def test_h1_graph_builder_creation(self):
        """Test that GraphBuilder can be created for H1."""
        from fast_td3.robots.graph_builder import GraphBuilder
        import torch
        
        builder = GraphBuilder(
            env_name="h1-stand-v0",
            batch_size=1,
            device=torch.device("cpu"),
            robot="h1"
        )
        self.assertIsNotNone(builder.robot)
        self.assertEqual(builder.robot.num_joints, 19)

    def test_g1_graph_builder_creation(self):
        """Test that GraphBuilder can be created for G1."""
        from fast_td3.robots.graph_builder import GraphBuilder
        import torch
        
        builder = GraphBuilder(
            env_name="g1-stand-v0",
            batch_size=1,
            device=torch.device("cpu"),
            robot="g1"
        )
        self.assertIsNotNone(builder.robot)
        self.assertEqual(builder.robot.num_joints, 37)

    def test_graph_builder_num_edges(self):
        """Test that GraphBuilder correctly reports number of edges."""
        from fast_td3.robots.graph_builder import GraphBuilder
        import torch
        
        builder_h1 = GraphBuilder(
            env_name="h1-stand-v0",
            batch_size=1,
            device=torch.device("cpu"),
            robot="h1"
        )
        
        builder_g1 = GraphBuilder(
            env_name="g1-stand-v0",
            batch_size=1,
            device=torch.device("cpu"),
            robot="g1"
        )
        
        self.assertEqual(builder_h1.num_edges, builder_h1.robot.num_edges)
        self.assertEqual(builder_g1.num_edges, builder_g1.robot.num_edges)


if __name__ == "__main__":
    unittest.main()
