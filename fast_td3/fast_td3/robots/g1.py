class G1:
    joint_dict = {
        # Left leg
        "left_hip_pitch": 0,
        "left_hip_roll": 1,
        "left_hip_yaw": 2,
        "left_knee": 3,
        "left_ankle_pitch": 4,
        "left_ankle_roll": 5,

        # Right leg
        "right_hip_pitch": 6,
        "right_hip_roll": 7,
        "right_hip_yaw": 8,
        "right_knee": 9,
        "right_ankle_pitch": 10,
        "right_ankle_roll": 11,

        # Torso
        "torso": 12,

        # Left arm
        "left_shoulder_pitch": 13,
        "left_shoulder_roll": 14,
        "left_shoulder_yaw": 15,
        "left_elbow_pitch": 16,
        "left_elbow_roll": 17,
        "left_zero": 18,
        "left_one": 19,
        "left_two": 20,
        "left_three": 21,
        "left_four": 22,
        "left_five": 23,
        "left_six": 24,

        # Right arm
        "right_shoulder_pitch": 25,
        "right_shoulder_roll": 26,
        "right_shoulder_yaw": 27,
        "right_elbow_pitch": 28,
        "right_elbow_roll": 29,
        "right_zero": 30,
        "right_one": 31,
        "right_two": 32,
        "right_three": 33,
        "right_four": 34,
        "right_five": 35,
        "right_six": 36,
    }

    edge_list = [
        # Left leg chain
        (joint_dict["left_hip_pitch"], joint_dict["left_hip_roll"]),
        (joint_dict["left_hip_roll"], joint_dict["left_hip_yaw"]),
        (joint_dict["left_hip_yaw"], joint_dict["left_knee"]),
        (joint_dict["left_knee"], joint_dict["left_ankle_pitch"]),
        (joint_dict["left_ankle_pitch"], joint_dict["left_ankle_roll"]),

        # Right leg chain
        (joint_dict["right_hip_pitch"], joint_dict["right_hip_roll"]),
        (joint_dict["right_hip_roll"], joint_dict["right_hip_yaw"]),
        (joint_dict["right_hip_yaw"], joint_dict["right_knee"]),
        (joint_dict["right_knee"], joint_dict["right_ankle_pitch"]),
        (joint_dict["right_ankle_pitch"], joint_dict["right_ankle_roll"]),

        # Torso
        (joint_dict["torso"], joint_dict["left_hip_pitch"]),
        (joint_dict["torso"], joint_dict["right_hip_pitch"]),
        (joint_dict["torso"], joint_dict["left_shoulder_pitch"]),
        (joint_dict["torso"], joint_dict["right_shoulder_pitch"]),

        # Left arm
        (joint_dict["left_shoulder_pitch"], joint_dict["left_shoulder_roll"]),
        (joint_dict["left_shoulder_roll"], joint_dict["left_shoulder_yaw"]),
        (joint_dict["left_shoulder_yaw"], joint_dict["left_elbow_pitch"]),
        (joint_dict["left_elbow_pitch"], joint_dict["left_elbow_roll"]),
        (joint_dict["left_elbow_roll"], joint_dict["left_zero"]),
        (joint_dict["left_zero"], joint_dict["left_one"]),
        (joint_dict["left_one"], joint_dict["left_two"]),
        (joint_dict["left_zero"], joint_dict["left_three"]),
        (joint_dict["left_three"], joint_dict["left_four"]),
        (joint_dict["left_three"], joint_dict["left_five"]),
        (joint_dict["left_five"], joint_dict["left_six"]),

        # Right arm
        (joint_dict["right_shoulder_pitch"], joint_dict["right_shoulder_roll"]),
        (joint_dict["right_shoulder_roll"], joint_dict["right_shoulder_yaw"]),
        (joint_dict["right_shoulder_yaw"], joint_dict["right_elbow_pitch"]),
        (joint_dict["right_elbow_pitch"], joint_dict["right_elbow_roll"]),
        (joint_dict["right_elbow_roll"], joint_dict["right_zero"]),
        (joint_dict["right_zero"], joint_dict["right_one"]),
        (joint_dict["right_one"], joint_dict["right_two"]),
        (joint_dict["right_zero"], joint_dict["right_three"]),
        (joint_dict["right_three"], joint_dict["right_four"]),
        (joint_dict["right_three"], joint_dict["right_five"]),
        (joint_dict["right_five"], joint_dict["right_six"]),
    ]

    edge_type_encoding = [
        # Left leg
        0, 1, 2, 3, 4,

        # Right leg
        0, 1, 2, 3, 4,

        # Torso connections
        5, 5, 6, 6,

        # Left arm
        7, 8, 9, 10, 11, 12, 13, 12, 14, 12, 15,

        # Right arm
        7, 8, 9, 10, 11, 12, 13, 12, 14, 12, 15,
    ]

    num_edges = len(edge_list)
    num_nodes = len(joint_dict)