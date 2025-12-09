import os
import sys
import json
import time
import uuid

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
import random
import tqdm
import wandb
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tensordict import TensorDict, from_module

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")
from fast_td3.fast_td3_utils import (
    SimpleReplayBufferGNN,
    DictEmpiricalNormalization,
    EmpiricalNormalization,
    save_params,
    unflatten_obs,
)
from fast_td3 import Critic

from fast_td3.train_utils import create_actor, collect_gradient_stats
from fast_td3.hyperparams import HumanoidBenchArgs
import argparse
from fast_td3.environments.humanoid_bench_env import HumanoidBenchEnv
from IPython.display import display, HTML
import base64
import imageio
import tempfile
import os


def main():
    parser = argparse.ArgumentParser(description="Train humanoid using FastTD3")
    parser.add_argument(
        "--actor",
        type=str,
        default="egnn_dict",
        help="The kind of actor to use.",
        choices=["egnn", "egnn_dict", "mlp"],
    )
    parser.add_argument("--env_name", type=str, default="h1-stand-v0")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=50000,
        help="Total number of timesteps to train for.",
    )
    parser.add_argument(
        "--render_interval",
        type=int,
        default=5000,
        help="Interval for rendering the environment.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5000,
        help="Interval for evaluating the agent.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments to use.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8192, help="Batch size for training."
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--no-wandb", dest="wandb", action="store_false", help="Disable wandb logging"
    )
    parser.set_defaults(wandb=True)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default=None,
        help="Additional model parameters (as defined in the class) in JSON format (path to the file)."
        "If not provided, defaults params will be used.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="",
        help="Description of the task/experiment to log to wandb.",
    )

    terminal_args = vars(parser.parse_args())

    if terminal_args["model_kwargs"] is not None:
        with open(terminal_args["model_kwargs"], "r") as f:
            model_kwargs = json.load(f)
    else:
        model_kwargs = {}

    args = HumanoidBenchArgs(
        env_name=terminal_args["env_name"],
        total_timesteps=terminal_args["total_timesteps"],
        render_interval=terminal_args["render_interval"],
        eval_interval=terminal_args["eval_interval"],
        num_envs=terminal_args["num_envs"],
        batch_size=terminal_args["batch_size"],
        model_kwargs=model_kwargs,
    )

    print(f"Training with args: {terminal_args}")

    use_wandb = terminal_args["wandb"]
    uid = uuid.uuid4().hex[:6]  # 6-char unique ID
    run_name = f"{terminal_args['actor']}_{args.env_name}_{args.num_envs}envs_{args.total_timesteps}steps_{uid}"
    if use_wandb:
        
        config = vars(args)
        config["actor"] = terminal_args["actor"]
        config["task_description"] = terminal_args["task_description"]
        
        wandb.init(
            entity="thuaduc24042001-technical-university-of-munich",
            project="FastTD3 - new",
            name=run_name,
            config=config,
            save_code=True,
            settings=wandb.Settings(
                _disable_stats=True,  # disables CPU/memory/disk/GPU monitoring
                _disable_meta=True,  # disables system metadata collection
            ),
        )
        
        wandb.save("fast_td3/fast_td3/actors/gnn/egnn.py")
        wandb.save("fast_td3/fast_td3/robots/H1.py")
        wandb.save("fast_td3/fast_td3/robots/graph_builder.py")



    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    env_type = "humanoid_bench"
    envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
    eval_envs = envs
    render_env = HumanoidBenchEnv(
        args.env_name, 1, render_mode="rgb_array", device=device
    )

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if type(envs.num_privileged_obs) == int
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    action_low, action_high = -1.0, 1.0

    # Define observation shapes for dict-based normalization
    # These shapes match the unflattened observation format
    obs_shapes = {
        "pelvis_position": (3,),
        "pelvis_quaternion": (4,),
        "pelvis_linear_velocity": (3,),
        "pelvis_angular_velocity": (3,),
        "joint_positions": (19,),  # For H1 robot - 19 body joints
        "joint_velocities": (19,),  # For H1 robot - 19 body joints
        "joint_x": (20, 3),  # For H1 robot - 20 joint anchors (3D coordinates), not normalized
    }

    if args.obs_normalization:
        # Use DictEmpiricalNormalization for dict observations
        # This normalizes each feature type separately
        obs_normalizer = DictEmpiricalNormalization(
            obs_shapes=obs_shapes, 
            device=device,
            skip_keys=["joint_x"]  # Don't normalize joint_x
        )
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    def normalize_obs(flat_obs, joint_x):
        """
        Normalize observations using DictEmpiricalNormalization.
        
        Args:
            flat_obs: Flat observation tensor from environment
            joint_x: Joint anchor coordinates
            
        Returns:
            Normalized dict observation
        """
        if isinstance(obs_normalizer, nn.Identity):
            # No normalization - just unflatten
            return unflatten_obs(flat_obs, joint_x)
        else:
            # Unflatten then normalize
            obs_dict = unflatten_obs(flat_obs, joint_x)
            return obs_normalizer(obs_dict)
    
    normalize_critic_obs = critic_obs_normalizer.forward

    # Create the main actor and actor detach (twin actor)
    actor = create_actor(
        actor_type=terminal_args["actor"],
        n_obs=n_obs,
        n_act=n_act,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        device=device,
        init_scale=args.init_scale,
        env_name=terminal_args["env_name"],
        model_kwargs=model_kwargs,
        actor_hidden_dim=args.actor_hidden_dim,
    )
    actor_detach = create_actor(
        actor_type=terminal_args["actor"],
        n_obs=n_obs,
        n_act=n_act,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        device=device,
        init_scale=args.init_scale,
        env_name=terminal_args["env_name"],
        model_kwargs=model_kwargs,
        actor_hidden_dim=args.actor_hidden_dim,
    )

    print(f"Actor num of parameters: {sum(p.numel() for p in actor.parameters())}")
    for name in actor.named_parameters():
        print(f"{name[0]} - {name[1].shape}")

    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    # critic
    qnet = Critic(
        n_obs=obs_flat_dim,
        n_act=n_act,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dim=args.critic_hidden_dim,
        device=device,
    )

    qnet_target = Critic(
        n_obs=obs_flat_dim,
        n_act=n_act,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dim=args.critic_hidden_dim,
        device=device,
    )
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=args.critic_learning_rate,
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=args.actor_learning_rate,
        weight_decay=args.weight_decay,
    )

    rb = SimpleReplayBufferGNN(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=obs_flat_dim,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        playground_mode=env_type == "mujoco_playground",
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
        env_name=terminal_args["env_name"],
    )

    checkpoint_path = terminal_args["checkpoint_path"]
    if checkpoint_path is not None:
        torch_checkpoint = torch.load(
            f"{checkpoint_path}", map_location=device, weights_only=False
        )

        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        if hasattr(obs_normalizer, "load_state_dict") and torch_checkpoint.get("obs_normalizer_state"):
            obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        if hasattr(critic_obs_normalizer, "load_state_dict") and torch_checkpoint.get("critic_obs_normalizer_state"):
            critic_obs_normalizer.load_state_dict(torch_checkpoint["critic_obs_normalizer_state"])
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    def evaluate():
        """
        Evaluates the trained actor network's performance on the environment.

        This function runs evaluation episodes using the deterministic actor policy
        (without exploration noise) to measure the agent's current performance.
        It collects episode returns and lengths across multiple parallel environments
        and returns the average metrics.

        Returns:
            tuple: (average_episode_return, average_episode_length)
                - average_episode_return: Mean cumulative reward across all evaluation episodes
                - average_episode_length: Mean number of steps across all evaluation episodes
        """
        obs_normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        # Reset environment - returns flat obs and joint_x separately
        obs, joint_x = eval_envs.reset()

        # Run for a fixed number of steps
        for _ in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                # Unflatten and normalize observations
                norm_obs_dict = normalize_obs(obs, joint_x)
                
                # For egnn_dict actor, pass dict observations directly (includes joint_x)
                if terminal_args["actor"] == "egnn_dict":
                    actions = actor(norm_obs_dict)
                else:
                    # For standard actors, use flat obs and joint_x separately
                    actions = actor(obs, joint_x)

            next_obs, rewards, dones, _, next_joint_x = eval_envs.step(actions.float())
            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs
            joint_x = next_joint_x

        obs_normalizer.train()
        return episode_returns.mean().item(), episode_lengths.mean().item()

    def render_with_rollout():
        obs_normalizer.eval()

        # Quick rollout for rendering
        if env_type == "humanoid_bench":
            obs, joint_x = render_env.reset()
            renders = [render_env.render()]
        elif env_type == "isaaclab":
            raise NotImplementedError(
                "We don't support rendering for IsaacLab environments"
            )
        else:
            obs, joint_x = render_env.reset()
            if hasattr(render_env, 'state'):
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
                renders = [render_env.state]
            else:
                renders = []

        for i in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                # Unflatten and normalize observations
                norm_obs_dict = normalize_obs(obs, joint_x)
                
                # For egnn_dict actor, pass dict observations directly (includes joint_x)
                if terminal_args["actor"] == "egnn_dict":
                    actions = actor(norm_obs_dict)
                else:
                    # For standard actors, use flat obs and joint_x separately
                    actions = actor(obs, joint_x)
                    
            next_obs, _, done, _, next_joint_x = render_env.step(actions.float())
            if env_type == "mujoco_playground":
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            if i % 2 == 0:
                if env_type == "humanoid_bench":
                    renders.append(render_env.render())
                else:
                    if hasattr(render_env, 'state'):
                        renders.append(render_env.state)
            if done.any():
                break
            obs = next_obs
            joint_x = next_joint_x

        if env_type == "mujoco_playground":
            renders = render_env.render_trajectory(renders)

        obs_normalizer.train()
        return renders

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def update_main(data, logs_dict):
        """
        TD3 Critic Update Function - Updates the twin Q-networks (critics).

        This function implements the core critic learning in TD3 algorithm with:
        1. Target Policy Smoothing: Adds clipped noise to target actions to reduce overestimation
        2. Clipped Double Q-Learning: Uses minimum of two Q-values to combat overestimation bias
        3. Distributional RL: Uses categorical distributions for Q-values (if enabled)

        The critics learn to estimate Q-values for state-action pairs using temporal difference learning.
        """
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            # Extract transition data from replay buffer
            # Dict observations are in observations_dict (unflattened and normalized)
            observations_dict = data["observations_dict"]
            next_observations_dict = data["next"]["observations_dict"]
            
            # Flat observations for critic
            flat_observations = data["observations"]
            flat_next_observations = data["next"]["observations"]

            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = flat_observations
                next_critic_observations = flat_next_observations

            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()

            # Determine bootstrap mask for value function targets
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            # TARGET POLICY SMOOTHING: Add clipped noise to target actions
            # This reduces overestimation by making the target policy less deterministic
            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            # Get next actions from actor
            # For egnn_dict, actor needs dict observations
            if terminal_args["actor"] == "egnn_dict":
                # Actor receives dict observations directly (includes joint_x)
                next_state_actions = (
                    actor(next_observations_dict) + clipped_noise
                ).clamp(action_low, action_high)
            else:
                # Standard actor uses flat obs + joint_x
                next_state_actions = (
                    actor(flat_next_observations, data["next"]["xanchors"]) + clipped_noise
                ).clamp(action_low, action_high)

            # Compute target Q-values using target networks (no gradients)
            with torch.no_grad():
                # Get distributional projections for both target Q-networks
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        args.gamma,
                    )
                )

                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)

                # CLIPPED DOUBLE Q-LEARNING: Use minimum Q-value to reduce overestimation
                if args.use_cdq:
                    # Choose the distribution corresponding to the lower Q-value
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    # Use both distributions separately
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            # Compute current Q-values for the actual state-action pairs
            qf1, qf2 = qnet(critic_observations, actions)

            # Compute distributional TD loss using cross-entropy
            # This trains the Q-networks to match the target distributions
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        # Perform gradient descent on critic networks
        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        
        # Gradient clipping to prevent exploding gradients
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            qnet.parameters(),
            max_norm=(args.max_grad_norm if args.max_grad_norm > 0 else float("inf")),
        )
        scaler.step(q_optimizer)
        scaler.update()

        # Log training metrics
        logs_dict["buffer_rewards"] = rewards.mean()
        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_pol(data, logs_dict):
        """
        TD3 Actor Update Function - Updates the main actor network (policy).

        This function implements delayed policy updates in TD3:
        1. Uses the main actor network (not actor_detach) for policy optimization
        2. Maximizes the Q-value estimated by the critic networks
        3. Updates less frequently than critics to ensure stable Q-value estimates

        The actor learns to select actions that maximize the expected Q-value,
        effectively learning the optimal policy through the actor-critic framework.
        """
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            # Extract observations from replay buffer data
            # Dict observations are in observations_dict (unflattened and normalized)
            observations_dict = data["observations_dict"]
            flat_observations = data["observations"]
            
            # Use appropriate observations based on environment setup
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
            else:
                critic_observations = flat_observations

            # Compute Q-values for current states with actions from the main actor
            # Note: This uses the main 'actor' network, not 'actor_detach'
            if terminal_args["actor"] == "egnn_dict":
                # Actor receives dict observations directly (includes joint_x)
                actor_actions = actor(observations_dict)
            else:
                # Standard actor uses flat obs + joint_x
                actor_actions = actor(flat_observations, data["xanchors"])
                
            qf1, qf2 = qnet(critic_observations, actor_actions)

            # Convert distributional Q-values to scalar estimates
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))

            # Policy objective: maximize expected Q-value
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0

            # Actor loss: negative Q-value (we want to maximize Q, so minimize -Q)
            actor_loss = -qf_value.mean()

        # Perform gradient ascent on actor network (gradient descent on negative Q-value)
        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        
        # Gradient clipping to prevent exploding gradients
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            max_norm=(args.max_grad_norm if args.max_grad_norm > 0 else float("inf")),
        )
        scaler.step(actor_optimizer)
        scaler.update()

        # Log training metrics
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    if args.compile:
        mode = None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)
        normalize_obs = torch.compile(normalize_obs, mode=mode)
        normalize_critic_obs = torch.compile(normalize_critic_obs, mode=mode)

    def frames_to_video_html(frames, fps=30):
        """
        Convert a list of numpy arrays to an HTML5 video element.

        Args:
            frames (list): List of numpy arrays representing video frames
            fps (int): Frames per second for the video

        Returns:
            HTML object containing the video element
        """
        # Create a temporary file to store the video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_filename = temp_file.name

        # Save frames as video
        imageio.mimsave(temp_filename, frames, fps=fps)

        # Read the video file and encode it to base64
        with open(temp_filename, "rb") as f:
            video_data = f.read()
        video_b64 = base64.b64encode(video_data).decode("utf-8")

        # Create HTML video element
        video_html = f"""
        <video width="640" height="480" controls>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """

        # Clean up the temporary file
        os.unlink(temp_filename)

        return HTML(video_html)

    def update_video_display(frames, fps=30):
        """
        Display video frames as an embedded HTML5 video element.

        Args:
            frames (list): List of numpy arrays representing video frames
            fps (int): Frames per second for the video
        """
        video_html = frames_to_video_html(frames, fps=fps)
        display(video_html)

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs, joint_x = envs.reset()  # obs is flat, joint_x is separate
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    dones = None

    while global_step < args.total_timesteps:
        logs_dict = TensorDict()  # Dictionary to store training metrics for this step

        # ACTION SELECTION PHASE
        # Use actor_detach (behavioral policy) with exploration noise for data collection
        # No gradients needed for action selection during environment interaction
        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            # obs and joint_x from env are flat vectors
            # Unflatten and normalize observations
            norm_obs_dict = normalize_obs(obs, joint_x)
            
            # For egnn_dict actor, pass normalized dict observations directly (includes joint_x)
            if terminal_args["actor"] == "egnn_dict":
                actions = policy(obs=norm_obs_dict, joint_x_param=None, dones=dones)
            else:
                # For standard egnn/mlp actors, use flat obs and joint_x separately
                # Using flat obs and joint_x from environment
                actions = policy(obs=obs, xanchor=joint_x, dones=dones)

        # ENVIRONMENT INTERACTION PHASE
        # Take actions in the environment and collect transition data
        next_obs, rewards, dones, infos, next_joint_x = envs.step(actions.float())
        truncations = infos["time_outs"]  # Episodes ended due to time limits

        # Extract privileged observations for critic if using asymmetric observations
        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]

        # TRANSITION DATA PREPARATION
        # Handle episode boundaries correctly - use 'raw' observations for terminal states
        # This ensures we store the actual final state, not the auto-reset state
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        true_next_xanchor = torch.where(
            dones[:, None, None] > 0, infos["observations"]["raw"].get("xanchor", next_joint_x), next_joint_x
        )
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        # Create transition tuple (s, a, r, s', done, truncated) for replay buffer
        # Store flat observations directly (no normalization at storage time)
        transition = TensorDict(
            {
                "observations": obs,
                "xanchors": joint_x,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "xanchors": true_next_xanchor,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )

        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs

        # UPDATE OBSERVATIONS FOR NEXT ITERATION
        obs = next_obs
        joint_x = next_joint_x
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        # REPLAY BUFFER STORAGE
        # Store the transition in the replay buffer for later sampling during training
        rb.extend(transition)

        # TRAINING PHASE
        # Only start training after collecting enough initial data (learning_starts)
        batch_size = args.batch_size // args.num_envs
        if global_step > args.learning_starts:
            # Perform multiple training updates per environment step for sample efficiency
            for i in range(args.num_updates):
                # Sample a batch of transitions from replay buffer
                data = rb.sample(batch_size)

                # Unflatten and normalize observations for actor/critic
                # Observations are stored as flat in replay buffer
                # We unflatten them and normalize with DictEmpiricalNormalization
                data["observations_dict"] = normalize_obs(data["observations"], data["xanchors"])
                data["next"]["observations_dict"] = normalize_obs(data["next"]["observations"], data["next"]["xanchors"])

                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(
                        data["critic_observations"]
                    )
                    data["next"]["critic_observations"] = normalize_critic_obs(
                        data["next"]["critic_observations"]
                    )

                # CRITIC UPDATE (Q-function learning)
                # Always update critics - they learn Q-values for state-action pairs
                logs_dict = update_main(data, logs_dict)

                # ACTOR UPDATE (Policy learning) - DELAYED UPDATES
                # TD3 uses delayed policy updates: update actor less frequently than critics
                # This ensures Q-values are more stable when training the policy
                if args.num_updates > 1:
                    # Multiple updates per step: update policy every policy_frequency updates
                    if i % args.policy_frequency == 1:
                        logs_dict = update_pol(data, logs_dict)
                else:
                    # Single update per step: update policy every policy_frequency steps
                    if global_step % args.policy_frequency == 0:
                        logs_dict = update_pol(data, logs_dict)

                # TARGET NETWORK SOFT UPDATE
                # Slowly update target networks using exponential moving average
                # This provides stable targets for Q-learning (prevents moving targets)
                for param, target_param in zip(
                    qnet.parameters(), qnet_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            # LOGGING AND EVALUATION PHASE
            # Periodically log training metrics, evaluate performance, and save models
            if global_step > 0 and global_step % 100 == 0:
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "buffer_rewards": logs_dict["buffer_rewards"].mean(),
                        "env_rewards": rewards.mean(),
                    }
                    
                    # Collect detailed gradient statistics for actor (every 500 steps to avoid overhead)
                    if use_wandb and global_step % 500 == 0:
                        grad_stats = collect_gradient_stats(actor, "actor")
                        logs.update(grad_stats)

                    # EVALUATION: Test current policy performance without exploration
                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        eval_avg_return, eval_avg_length = evaluate()
                        # Reset training environments after evaluation (environment-specific hack)
                        if env_type in ["humanoid_bench_dict", "humanoid_bench", "isaaclab"]:
                            obs = envs.reset()
                        logs["eval_avg_return"] = eval_avg_return
                        logs["eval_avg_length"] = eval_avg_length

                    # RENDERING: Generate and display videos of current policy
                    if (
                        args.render_interval > 0
                        and global_step % args.render_interval == 0
                    ):
                        renders = render_with_rollout()
                        print_logs = {
                            k: v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in logs.items()
                        }
                        for k, v in print_logs.items():
                            print(f"{k}: {v:.4f}")
                        # Display video in notebook
                        update_video_display(renders, fps=30)
                        if use_wandb:
                            wandb.log(
                                {
                                    "render_video": wandb.Video(
                                        np.array(renders).transpose(
                                            0, 3, 1, 2
                                        ),  # Convert to (T, C, H, W) format
                                        fps=30,
                                        format="gif",
                                    )
                                },
                                step=global_step,
                            )

                if use_wandb:
                    wandb.log(
                        {
                            **logs,
                        },
                        step=global_step,
                    )

            if (
                args.save_interval > 0
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    critic_obs_normalizer,
                    args,
                    f"models/{run_name}_{global_step}.pt",
                )

        global_step += 1
        pbar.update(1)

    save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        f"models/{run_name}_final.pt",
    )


if __name__ == "__main__":
    main()
