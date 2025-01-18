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

# Function to start training
start_training() {
    print_color "32" "Starting training process..."
    python -m tdmpc2.train \
    disable_wandb=true \
    exp_name=tdmpc \
    task=humanoid_${TASK} \
    seed=0 \
    policy_path=${POLICY_PATH} \
    mean_path=${MEAN_PATH} \
    var_path=${VAR_PATH} \
    policy_type="reach_single" \
    batch_size=256 \
    eval_freq=50000 \
    save_csv=true \
    save_agent=true \
    save_video=false      # Explicitly disable video saving
}

# Start training and monitoring
start_training