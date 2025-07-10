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

    edge_list = [
        # Left leg - hip joints fully connected to each other
        (joint_dict["left_hip_yaw"], joint_dict["left_hip_roll"]),
        (joint_dict["left_hip_yaw"], joint_dict["left_hip_pitch"]),
        (joint_dict["left_hip_roll"], joint_dict["left_hip_pitch"]),
        # Left leg - all hip joints connect to knee
        (joint_dict["left_hip_yaw"], joint_dict["left_knee"]),
        (joint_dict["left_hip_roll"], joint_dict["left_knee"]),
        (joint_dict["left_hip_pitch"], joint_dict["left_knee"]),
        (joint_dict["left_knee"], joint_dict["left_ankle"]),
        # Right leg - hip joints fully connected to each other
        (joint_dict["right_hip_yaw"], joint_dict["right_hip_roll"]),
        (joint_dict["right_hip_yaw"], joint_dict["right_hip_pitch"]),
        (joint_dict["right_hip_roll"], joint_dict["right_hip_pitch"]),
        # Right leg - all hip joints connect to knee
        (joint_dict["right_hip_yaw"], joint_dict["right_knee"]),
        (joint_dict["right_hip_roll"], joint_dict["right_knee"]),
        (joint_dict["right_hip_pitch"], joint_dict["right_knee"]),
        (joint_dict["right_knee"], joint_dict["right_ankle"]),
        # Torso connections to hips
        (joint_dict["torso"], joint_dict["left_hip_yaw"]),
        (joint_dict["torso"], joint_dict["right_hip_yaw"]),
        # Left arm - shoulder joints fully connected to each other
        (joint_dict["left_shoulder_pitch"], joint_dict["left_shoulder_roll"]),
        (joint_dict["left_shoulder_pitch"], joint_dict["left_shoulder_yaw"]),
        (joint_dict["left_shoulder_roll"], joint_dict["left_shoulder_yaw"]),
        # Left arm - all shoulder joints connect to torso
        (joint_dict["torso"], joint_dict["left_shoulder_pitch"]),
        (joint_dict["torso"], joint_dict["left_shoulder_roll"]),
        (joint_dict["torso"], joint_dict["left_shoulder_yaw"]),
        (joint_dict["left_shoulder_yaw"], joint_dict["left_elbow"]),
        # Right arm - shoulder joints fully connected to each other
        (joint_dict["right_shoulder_pitch"], joint_dict["right_shoulder_roll"]),
        (joint_dict["right_shoulder_pitch"], joint_dict["right_shoulder_yaw"]),
        (joint_dict["right_shoulder_roll"], joint_dict["right_shoulder_yaw"]),
        # Right arm - all shoulder joints connect to torso
        (joint_dict["torso"], joint_dict["right_shoulder_pitch"]),
        (joint_dict["torso"], joint_dict["right_shoulder_roll"]),
        (joint_dict["torso"], joint_dict["right_shoulder_yaw"]),
        (joint_dict["right_shoulder_yaw"], joint_dict["right_elbow"]),
    ]

    edge_type_encoding = [  
        # Left hip joints connected to each other
        0,  # (left_hip_yaw, left_hip_roll)       - left_hip_connection
        0,  # (left_hip_yaw, left_hip_pitch)      - left_hip_connection
        0,  # (left_hip_roll, left_hip_pitch)     - left_hip_connection
        # Left hip joints to knee
        1,  # (left_hip_yaw, left_knee)           - left_hip_to_knee
        1,  # (left_hip_roll, left_knee)          - left_hip_to_knee
        1,  # (left_hip_pitch, left_knee)         - left_hip_to_knee
        2,  # (left_knee, left_ankle)             - left_leg

        # Right hip joints connected to each other
        0,  # (right_hip_yaw, right_hip_roll)     - right_hip_connection
        0,  # (right_hip_yaw, right_hip_pitch)    - right_hip_connection
        0,  # (right_hip_roll, right_hip_pitch)   - right_hip_connection
        # Right hip joints to knee
        1,  # (right_hip_yaw, right_knee)         - right_hip_to_knee
        1,  # (right_hip_roll, right_knee)        - right_hip_to_knee
        1,  # (right_hip_pitch, right_knee)       - right_hip_to_knee
        2,  # (right_knee, right_ankle)           - right_leg

        # Torso to hips
        3,  # (torso, left_hip_yaw)               - torso_to_hip
        3,  # (torso, right_hip_yaw)              - torso_to_hip

        # Left shoulder joints connected to each other
        4,  # (left_shoulder_pitch, left_shoulder_roll)   - left_shoulder_connection
        4,  # (left_shoulder_pitch, left_shoulder_yaw)    - left_shoulder_connection
        4, # (left_shoulder_roll, left_shoulder_yaw)     - left_shoulder_connection
        # Left shoulder joints to torso
        5, # (torso, left_shoulder_pitch)        - torso_to_shoulder
        5, # (torso, left_shoulder_roll)         - torso_to_shoulder
        5, # (torso, left_shoulder_yaw)          - torso_to_shoulder
        6, # (left_shoulder_yaw, left_elbow)     - left_arm

        # Right shoulder joints connected to each other
        4,  # (right_shoulder_pitch, right_shoulder_roll) - right_shoulder_connection
        4,  # (right_shoulder_pitch, right_shoulder_yaw)  - right_shoulder_connection
        4, # (right_shoulder_roll, right_shoulder_yaw)   - right_shoulder_connection
        # Right shoulder joints to torso
        5, # (torso, right_shoulder_pitch)       - torso_to_shoulder
        5, # (torso, right_shoulder_roll)        - torso_to_shoulder
        5, # (torso, right_shoulder_yaw)         - torso_to_shoulder
        6, # (right_shoulder_yaw, right_elbow)   - right_arm
    ]

    num_edges = len(edge_list)
    num_nodes = len(joint_dict)