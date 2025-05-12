#!/bin/bash
export WORK_DIR="$(pwd)"
export BASE_DIR="$WORK_DIR"
export TASK="h1-push-v0"
export POLICY_PATH="${BASE_DIR}/data/reach_one_hand/torch_model.pt"
export MEAN_PATH="${BASE_DIR}/data/reach_one_hand/mean.npy"
export VAR_PATH="${BASE_DIR}/data/reach_one_hand/var.npy"
export HIGH_LEVEL_POLICY="${BASE_DIR}/logs/humanoid_h1hand-push-v0/0/tdmpc/models/1800154.pt"

python -m inference \
    --env ${TASK} \
    --policy_type reach_single \
    --policy_path ${POLICY_PATH} \
    --mean_path ${MEAN_PATH} \
    --var_path ${VAR_PATH} \
    --high_level_policy_path ${HIGH_LEVEL_POLICY}