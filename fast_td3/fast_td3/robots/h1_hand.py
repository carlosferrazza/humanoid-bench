class H1Hand:

    joint_dict = {
        # Base joints (excluding free_base as it's not a controllable joint)
        "left_hip_yaw": 1,
        "left_hip_roll": 2,
        "left_hip_pitch": 3,
        "left_knee": 4,
        "left_ankle": 5,
        "right_hip_yaw": 6,
        "right_hip_roll": 7,
        "right_hip_pitch": 8,
        "right_knee": 9,
        "right_ankle": 10,
        "torso": 11,
        "left_shoulder_pitch": 12,
        "left_shoulder_roll": 13,
        "left_shoulder_yaw": 14,
        "left_elbow": 15,
        "left_wrist_yaw": 16,
        # Left hand joints
        "lh_WRJ2": 17,
        "lh_WRJ1": 18,
        "lh_FFJ4": 19,
        "lh_FFJ3": 20,
        "lh_FFJ2": 21,
        "lh_FFJ1": 22,
        "lh_MFJ4": 23,
        "lh_MFJ3": 24,
        "lh_MFJ2": 25,
        "lh_MFJ1": 26,
        "lh_RFJ4": 27,
        "lh_RFJ3": 28,
        "lh_RFJ2": 29,
        "lh_RFJ1": 30,
        "lh_LFJ5": 31,
        "lh_LFJ4": 32,
        "lh_LFJ3": 33,
        "lh_LFJ2": 34,
        "lh_LFJ1": 35,
        "lh_THJ5": 36,
        "lh_THJ4": 37,
        "lh_THJ3": 38,
        "lh_THJ2": 39,
        "lh_THJ1": 40,
        # Right arm joints
        "right_shoulder_pitch": 41,
        "right_shoulder_roll": 42,
        "right_shoulder_yaw": 43,
        "right_elbow": 44,
        "right_wrist_yaw": 45,
        # Right hand joints
        "rh_WRJ2": 46,
        "rh_WRJ1": 47,
        "rh_FFJ4": 48,
        "rh_FFJ3": 49,
        "rh_FFJ2": 50,
        "rh_FFJ1": 51,
        "rh_MFJ4": 52,
        "rh_MFJ3": 53,
        "rh_MFJ2": 54,
        "rh_MFJ1": 55,
        "rh_RFJ4": 56,
        "rh_RFJ3": 57,
        "rh_RFJ2": 58,
        "rh_RFJ1": 59,
        "rh_LFJ5": 60,
        "rh_LFJ4": 61,
        "rh_LFJ3": 62,
        "rh_LFJ2": 63,
        "rh_LFJ1": 64,
        "rh_THJ5": 65,
        "rh_THJ4": 66,
        "rh_THJ3": 67,
        "rh_THJ2": 68,
        "rh_THJ1": 69,
    }

    edge_list = [
        # Left leg
        (joint_dict["left_hip_yaw"], joint_dict["left_hip_roll"]),
        (joint_dict["left_hip_roll"], joint_dict["left_hip_pitch"]),
        (joint_dict["left_hip_pitch"], joint_dict["left_knee"]),
        (joint_dict["left_knee"], joint_dict["left_ankle"]),
        # Right leg
        (joint_dict["right_hip_yaw"], joint_dict["right_hip_roll"]),
        (joint_dict["right_hip_roll"], joint_dict["right_hip_pitch"]),
        (joint_dict["right_hip_pitch"], joint_dict["right_knee"]),
        (joint_dict["right_knee"], joint_dict["right_ankle"]),
        # Torso connections
        (joint_dict["torso"], joint_dict["left_hip_yaw"]),
        (joint_dict["torso"], joint_dict["right_hip_yaw"]),
        # Left arm
        (joint_dict["torso"], joint_dict["left_shoulder_pitch"]),
        (joint_dict["left_shoulder_pitch"], joint_dict["left_shoulder_roll"]),
        (joint_dict["left_shoulder_roll"], joint_dict["left_shoulder_yaw"]),
        (joint_dict["left_shoulder_yaw"], joint_dict["left_elbow"]),
        (joint_dict["left_elbow"], joint_dict["left_wrist_yaw"]),
        # Left hand - wrist connections
        (joint_dict["left_wrist_yaw"], joint_dict["lh_WRJ2"]),
        (joint_dict["lh_WRJ2"], joint_dict["lh_WRJ1"]),
        # Left hand - index finger (FF)
        (joint_dict["lh_WRJ1"], joint_dict["lh_FFJ4"]),
        (joint_dict["lh_FFJ4"], joint_dict["lh_FFJ3"]),
        (joint_dict["lh_FFJ3"], joint_dict["lh_FFJ2"]),
        (joint_dict["lh_FFJ2"], joint_dict["lh_FFJ1"]),
        # Left hand - middle finger (MF)
        (joint_dict["lh_WRJ1"], joint_dict["lh_MFJ4"]),
        (joint_dict["lh_MFJ4"], joint_dict["lh_MFJ3"]),
        (joint_dict["lh_MFJ3"], joint_dict["lh_MFJ2"]),
        (joint_dict["lh_MFJ2"], joint_dict["lh_MFJ1"]),
        # Left hand - ring finger (RF)
        (joint_dict["lh_WRJ1"], joint_dict["lh_RFJ4"]),
        (joint_dict["lh_RFJ4"], joint_dict["lh_RFJ3"]),
        (joint_dict["lh_RFJ3"], joint_dict["lh_RFJ2"]),
        (joint_dict["lh_RFJ2"], joint_dict["lh_RFJ1"]),
        # Left hand - little finger (LF)
        (joint_dict["lh_WRJ1"], joint_dict["lh_LFJ5"]),
        (joint_dict["lh_LFJ5"], joint_dict["lh_LFJ4"]),
        (joint_dict["lh_LFJ4"], joint_dict["lh_LFJ3"]),
        (joint_dict["lh_LFJ3"], joint_dict["lh_LFJ2"]),
        (joint_dict["lh_LFJ2"], joint_dict["lh_LFJ1"]),
        # Left hand - thumb (TH)
        (joint_dict["lh_WRJ1"], joint_dict["lh_THJ5"]),
        (joint_dict["lh_THJ5"], joint_dict["lh_THJ4"]),
        (joint_dict["lh_THJ4"], joint_dict["lh_THJ3"]),
        (joint_dict["lh_THJ3"], joint_dict["lh_THJ2"]),
        (joint_dict["lh_THJ2"], joint_dict["lh_THJ1"]),
        # Right arm
        (joint_dict["torso"], joint_dict["right_shoulder_pitch"]),
        (joint_dict["right_shoulder_pitch"], joint_dict["right_shoulder_roll"]),
        (joint_dict["right_shoulder_roll"], joint_dict["right_shoulder_yaw"]),
        (joint_dict["right_shoulder_yaw"], joint_dict["right_elbow"]),
        (joint_dict["right_elbow"], joint_dict["right_wrist_yaw"]),
        # Right hand - wrist connections
        (joint_dict["right_wrist_yaw"], joint_dict["rh_WRJ2"]),
        (joint_dict["rh_WRJ2"], joint_dict["rh_WRJ1"]),
        # Right hand - index finger (FF)
        (joint_dict["rh_WRJ1"], joint_dict["rh_FFJ4"]),
        (joint_dict["rh_FFJ4"], joint_dict["rh_FFJ3"]),
        (joint_dict["rh_FFJ3"], joint_dict["rh_FFJ2"]),
        (joint_dict["rh_FFJ2"], joint_dict["rh_FFJ1"]),
        # Right hand - middle finger (MF)
        (joint_dict["rh_WRJ1"], joint_dict["rh_MFJ4"]),
        (joint_dict["rh_MFJ4"], joint_dict["rh_MFJ3"]),
        (joint_dict["rh_MFJ3"], joint_dict["rh_MFJ2"]),
        (joint_dict["rh_MFJ2"], joint_dict["rh_MFJ1"]),
        # Right hand - ring finger (RF)
        (joint_dict["rh_WRJ1"], joint_dict["rh_RFJ4"]),
        (joint_dict["rh_RFJ4"], joint_dict["rh_RFJ3"]),
        (joint_dict["rh_RFJ3"], joint_dict["rh_RFJ2"]),
        (joint_dict["rh_RFJ2"], joint_dict["rh_RFJ1"]),
        # Right hand - little finger (LF)
        (joint_dict["rh_WRJ1"], joint_dict["rh_LFJ5"]),
        (joint_dict["rh_LFJ5"], joint_dict["rh_LFJ4"]),
        (joint_dict["rh_LFJ4"], joint_dict["rh_LFJ3"]),
        (joint_dict["rh_LFJ3"], joint_dict["rh_LFJ2"]),
        (joint_dict["rh_LFJ2"], joint_dict["rh_LFJ1"]),
        # Right hand - thumb (TH)
        (joint_dict["rh_WRJ1"], joint_dict["rh_THJ5"]),
        (joint_dict["rh_THJ5"], joint_dict["rh_THJ4"]),
        (joint_dict["rh_THJ4"], joint_dict["rh_THJ3"]),
        (joint_dict["rh_THJ3"], joint_dict["rh_THJ2"]),
        (joint_dict["rh_THJ2"], joint_dict["rh_THJ1"]),
    ]

    edge_type_encoding = [
        # Left leg
        0,  # (left_hip_yaw, left_hip_roll)      - left_leg
        1,  # (left_hip_roll, left_hip_pitch)    - left_leg
        2,  # (left_hip_pitch, left_knee)        - left_leg
        3,  # (left_knee, left_ankle)            - left_leg
        # Right leg
        0,  # (right_hip_yaw, right_hip_roll)    - right_leg
        1,  # (right_hip_roll, right_hip_pitch)  - right_leg
        2,  # (right_hip_pitch, right_knee)      - right_leg
        3,  # (right_knee, right_ankle)          - right_leg
        # Torso connections
        4,  # (torso, left_hip_yaw)              - torso_connection
        4,  # (torso, right_hip_yaw)             - torso_connection
        # Left arm
        5,  # (torso, left_shoulder_pitch)       - left_arm
        6,  # (left_shoulder_pitch, left_shoulder_roll) - left_arm
        7,  # (left_shoulder_roll, left_shoulder_yaw)   - left_arm
        8,  # (left_shoulder_yaw, left_elbow)           - left_arm
        9,  # (left_elbow, left_wrist_yaw)              - left_arm
        # Left hand - wrist connections
        10,  # (left_wrist_yaw, lh_WRJ2)          - left_hand_wrist
        11,  # (lh_WRJ2, lh_WRJ1)                 - left_hand_wrist
        # Left hand - index finger (FF)
        12,  # (lh_WRJ1, lh_FFJ4)                 - left_hand_index
        13,  # (lh_FFJ4, lh_FFJ3)                 - left_hand_index
        14,  # (lh_FFJ3, lh_FFJ2)                 - left_hand_index
        15,  # (lh_FFJ2, lh_FFJ1)                 - left_hand_index
        # Left hand - middle finger (MF)
        16,  # (lh_WRJ1, lh_MFJ4)                 - left_hand_middle
        17,  # (lh_MFJ4, lh_MFJ3)                 - left_hand_middle
        18,  # (lh_MFJ3, lh_MFJ2)                 - left_hand_middle
        19,  # (lh_MFJ2, lh_MFJ1)                 - left_hand_middle
        # Left hand - ring finger (RF)
        20,  # (lh_WRJ1, lh_RFJ4)                 - left_hand_ring
        21,  # (lh_RFJ4, lh_RFJ3)                 - left_hand_ring
        22,  # (lh_RFJ3, lh_RFJ2)                 - left_hand_ring
        23,  # (lh_RFJ2, lh_RFJ1)                 - left_hand_ring
        # Left hand - little finger (LF)
        24,  # (lh_WRJ1, lh_LFJ5)                 - left_hand_little
        25,  # (lh_LFJ5, lh_LFJ4)                 - left_hand_little
        26,  # (lh_LFJ4, lh_LFJ3)                 - left_hand_little
        27,  # (lh_LFJ3, lh_LFJ2)                 - left_hand_little
        28,  # (lh_LFJ2, lh_LFJ1)                 - left_hand_little
        # Left hand - thumb (TH)
        29,  # (lh_WRJ1, lh_THJ5)                 - left_hand_thumb
        30,  # (lh_THJ5, lh_THJ4)                 - left_hand_thumb
        31,  # (lh_THJ4, lh_THJ3)                 - left_hand_thumb
        32,  # (lh_THJ3, lh_THJ2)                 - left_hand_thumb
        33,  # (lh_THJ2, lh_THJ1)                 - left_hand_thumb
        # Right arm
        5,  # (torso, right_shoulder_pitch)      - right_arm
        6,  # (right_shoulder_pitch, right_shoulder_roll) - right_arm
        7,  # (right_shoulder_roll, right_shoulder_yaw)   - right_arm
        8,  # (right_shoulder_yaw, right_elbow)           - right_arm
        9,  # (right_elbow, right_wrist_yaw)              - right_arm
        # Right hand - wrist connections
        34,  # (right_wrist_yaw, rh_WRJ2)         - right_hand_wrist
        35,  # (rh_WRJ2, rh_WRJ1)                 - right_hand_wrist
        # Right hand - index finger (FF)
        36,  # (rh_WRJ1, rh_FFJ4)                 - right_hand_index
        37,  # (rh_FFJ4, rh_FFJ3)                 - right_hand_index
        38,  # (rh_FFJ3, rh_FFJ2)                 - right_hand_index
        39,  # (rh_FFJ2, rh_FFJ1)                 - right_hand_index
        # Right hand - middle finger (MF)
        40,  # (rh_WRJ1, rh_MFJ4)                 - right_hand_middle
        41,  # (rh_MFJ4, rh_MFJ3)                 - right_hand_middle
        42,  # (rh_MFJ3, rh_MFJ2)                 - right_hand_middle
        43,  # (rh_MFJ2, rh_MFJ1)                 - right_hand_middle
        # Right hand - ring finger (RF)
        44,  # (rh_WRJ1, rh_RFJ4)                 - right_hand_ring
        45,  # (rh_RFJ4, rh_RFJ3)                 - right_hand_ring
        46,  # (rh_RFJ3, rh_RFJ2)                 - right_hand_ring
        47,  # (rh_RFJ2, rh_RFJ1)                 - right_hand_ring
        # Right hand - little finger (LF)
        48,  # (rh_WRJ1, rh_LFJ5)                 - right_hand_little
        49,  # (rh_LFJ5, rh_LFJ4)                 - right_hand_little
        50,  # (rh_LFJ4, rh_LFJ3)                 - right_hand_little
        51,  # (rh_LFJ3, rh_LFJ2)                 - right_hand_little
        52,  # (rh_LFJ2, rh_LFJ1)                 - right_hand_little
        # Right hand - thumb (TH)
        53,  # (rh_WRJ1, rh_THJ5)                 - right_hand_thumb
        54,  # (rh_THJ5, rh_THJ4)                 - right_hand_thumb
        55,  # (rh_THJ4, rh_THJ3)                 - right_hand_thumb
        56,  # (rh_THJ3, rh_THJ2)                 - right_hand_thumb
        57,  # (rh_THJ2, rh_THJ1)                 - right_hand_thumb
    ]
