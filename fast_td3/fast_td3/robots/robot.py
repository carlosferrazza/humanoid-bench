import enum
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class Robot(ABC):
    """Abstract base class for all robot models.
    
    This class defines the common interface that all robot implementations
    should follow, including joint definitions, connections, and layout methods.
    """
    
    JOINT = None  # To be defined in subclasses as an enum.IntEnum

    @property
    def joint_connections(self) -> List[Tuple[int, int]]:
        """List of tuples defining direct connections between joints.
        
        Each tuple contains two joint IDs that are directly connected.
        """
        return []
    
    @property
    def connection_colors(self) -> Dict[str, str]:
        """Define colors for different types of joint connections.
        
        Returns:
            A dictionary mapping connection types to hex color codes
        """
        return {}
    
    @abstractmethod
    def get_joint_name(self, joint_id: int) -> str:
        """Convert joint ID to readable name.
        
        Args:
            joint_id: The integer ID of the joint
            
        Returns:
            A human-readable name for the joint
        """
        pass
    
    @abstractmethod
    def get_robot_layout_positions(self) -> Dict[int, Tuple[float, float]]:
        """Define positions to create a robot-like layout.
        
        Returns:
            A dictionary mapping joint IDs to their (x, y) positions
            for visualization purposes
        """
        pass
    
    @abstractmethod
    def get_connection_type(self, joint1_id: int, joint2_id: int) -> str:
        """Classify the type of connection between two joints.
        
        Args:
            joint1_id: ID of the first joint
            joint2_id: ID of the second joint
            
        Returns:
            A string describing the connection type (e.g., "torso_hip", "left_leg_other")
        """
        pass
    
    def get_joint_count(self) -> int:
        """Get the total number of joints in the robot.
        
        Returns:
            The number of joints defined in the JOINT enum
        """
        if self.JOINT is None:
            return 0
        return len(self.JOINT)
    
    def get_all_joint_names(self) -> List[str]:
        """Get names of all joints in the robot.
        
        Returns:
            A list of all joint names
        """
        if self.JOINT is None:
            return []
        return [self.get_joint_name(joint_id) for joint_id in self.JOINT]
    
    def get_connection_count(self) -> int:
        """Get the total number of joint connections.
        
        Returns:
            The number of connections defined in joint_connections
        """
        return len(self.joint_connections)
    
    def is_joint_connected(self, joint1_id: int, joint2_id: int) -> bool:
        """Check if two joints are directly connected.
        
        Args:
            joint1_id: ID of the first joint
            joint2_id: ID of the second joint
            
        Returns:
            True if the joints are connected, False otherwise
        """
        return ((joint1_id, joint2_id) in self.joint_connections or 
                (joint2_id, joint1_id) in self.joint_connections)
    
    def get_connected_joints(self, joint_id: int) -> List[int]:
        """Get all joints connected to a given joint.
        
        Args:
            joint_id: ID of the joint to find connections for
            
        Returns:
            A list of joint IDs that are connected to the given joint
        """
        connected = []
        for connection in self.joint_connections:
            if connection[0] == joint_id:
                connected.append(connection[1])
            elif connection[1] == joint_id:
                connected.append(connection[0])
        return list(set(connected))  # Remove duplicates
    
    def validate_connections(self) -> List[str]:
        """Validate that all connections reference valid joints.
        
        Returns:
            A list of error messages for invalid connections, empty if all valid
        """
        if self.JOINT is None:
            return ["JOINT enum not defined"]
        
        errors = []
        valid_joints = set(self.JOINT)
        
        for i, connection in enumerate(self.joint_connections):
            if len(connection) != 2:
                errors.append(f"Connection {i}: Expected tuple of 2 joints, got {len(connection)}")
                continue
                
            joint1, joint2 = connection
            if joint1 not in valid_joints:
                errors.append(f"Connection {i}: Invalid joint {joint1}")
            if joint2 not in valid_joints:
                errors.append(f"Connection {i}: Invalid joint {joint2}")
                
        return errors
    
    def __str__(self) -> str:
        """String representation of the robot."""
        return f"{self.__class__.__name__}(joints={self.get_joint_count()}, connections={self.get_connection_count()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the robot."""
        return self.__str__()