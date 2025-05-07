#!/bin/bash

# Function to print colored output
print_color() {
    local color=$1
    local text=$2
    echo -e "\033[${color}m${text}\033[0m"
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        print_color "31" "Error: File $1 does not exist!"
        exit 1
    fi
}

# Initialize conda for bash
print_color "36" "Initializing conda..."
CONDA_PATH="/afs/cs.stanford.edu/u/ywlin/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_PATH"

# Activate conda environment
print_color "36" "Activating humanoidbench environment..."
conda activate humanoidbench

# Change to correct working directory
WORK_DIR="/viscam/u/ywlin/dp3-humanoidBench/humanoid-bench"
if [ ! -d "$WORK_DIR" ]; then
    print_color "31" "Error: Working directory $WORK_DIR does not exist!"
    exit 1
fi
cd "$WORK_DIR"
print_color "36" "Changed working directory to: $(pwd)"

# Set up environment variables
print_color "36" "Setting up environment variables..."
export MUJOCO_GL="egl"
export BASE_DIR="$WORK_DIR"
export TASK="h1hand-push-v0"
export POLICY_PATH="${BASE_DIR}/data/reach_one_hand/torch_model.pt"
export MEAN_PATH="${BASE_DIR}/data/reach_one_hand/mean.npy"
export VAR_PATH="${BASE_DIR}/data/reach_one_hand/var.npy"

# Check if required files exist
print_color "36" "Checking required files..."
check_file "$POLICY_PATH"
check_file "$MEAN_PATH"
check_file "$VAR_PATH"

# Create log directory
LOG_DIR="logs/${TASK}/0/dreamer"
mkdir -p "$LOG_DIR"

# Function to start training
start_training() {
    print_color "32" "Starting training process..."
    python -m embodied.agents.dreamerv3.train \
        --configs humanoid_benchmark \
        --run.wandb False \
        --method dreamer_${TASK}_hierarchical \
        --logdir ${LOG_DIR} \
        --env.humanoid.policy_path ${POLICY_PATH} \
        --env.humanoid.mean_path ${MEAN_PATH} \
        --env.humanoid.var_path ${VAR_PATH} \
        --env.humanoid.policy_type="reach_single" \
        --task humanoid_${TASK} \
        --seed 0 \
        > ${LOG_DIR}/training.log 2>&1
}

# Function to start monitoring
print_monitoring_commands() {
    print_color "33" "To monitor training, open new terminal windows and run these commands:"
    print_color "33" "  tail -f ${LOG_DIR}/training.log"
    print_color "33" "  watch -n 60 'ls -ltr ${LOG_DIR}/models/'"
    print_color "33" "  tail -f ${LOG_DIR}/metrics.jsonl"
}

# Start training and monitoring
print_monitoring_commands
start_training