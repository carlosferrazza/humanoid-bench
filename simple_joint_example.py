#!/usr/bin/env python3
"""
Simple example showing how to find joint types in HumanoidBench
"""

import mujoco
import os

def get_joint_type_info(robot_name="h1"):
    """
    Simple function to get joint type information for a HumanoidBench robot
    """
    # Load the robot model
    asset_path = os.path.join(os.path.dirname(__file__), "humanoid_bench", "assets")
    model_path = os.path.join(asset_path, "robots", f"{robot_name}_pos.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    
    # Joint type mapping
    joint_types = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    
    # Get joint information
    joint_info = {}
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type_num = model.jnt_type[i]
        joint_type_name = joint_types.get(joint_type_num, f"unknown({joint_type_num})")
        joint_axis = model.jnt_axis[i]
        joint_range = model.jnt_range[i] if model.jnt_limited[i] else None
        
        joint_info[joint_name] = {
            'type_num': joint_type_num,
            'type_name': joint_type_name,
            'axis': joint_axis.copy(),
            'range': joint_range.copy() if joint_range is not None else None,
            'limited': bool(model.jnt_limited[i])
        }
    
    return joint_info, model

# Example usage:
if __name__ == "__main__":
    # Get joint info for H1 robot
    joint_info, model = get_joint_type_info("h1")
    
    # Print some examples
    print("Joint type examples:")
    for joint_name in ['free_base', 'left_hip_yaw', 'left_knee', 'torso']:
        if joint_name in joint_info:
            info = joint_info[joint_name]
            print(f"{joint_name}: {info['type_name']} joint, axis={info['axis']}")
    
    # Find all hinge joints
    print("\nAll hinge joints:")
    hinge_joints = [name for name, info in joint_info.items() if info['type_name'] == 'hinge']
    print(hinge_joints)
    
    # Get joint type by name (programmatically)
    def get_joint_type_by_name(joint_name, model):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            return model.jnt_type[joint_id]
        return None
    
    # Example: Check if a joint is a hinge joint
    joint_name = "left_hip_yaw"
    joint_type = get_joint_type_by_name(joint_name, model)
    is_hinge = joint_type == 3  # 3 = hinge
    print(f"\n{joint_name} is a hinge joint: {is_hinge}")
