import argparse
import pathlib
import os

import cv2
import gymnasium as gym
import torch
import numpy as np
from termcolor import colored

import humanoid_bench
from humanoid_bench.env import ROBOTS, TASKS
from tdmpc2.model_loader import get_agent, load_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HumanoidBench environment test")
    parser.add_argument("--env", help="e.g. h1-walk-v0")
    parser.add_argument("--keyframe", default=None)
    parser.add_argument("--high_level_policy_path", default=None) # added this line to replace random sampling with high-level policy
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--mean_path", default=None)
    parser.add_argument("--var_path", default=None)
    parser.add_argument("--policy_type", default=None)
    parser.add_argument("--blocked_hands", default="False")
    parser.add_argument("--small_obs", default="False")
    parser.add_argument("--obs_wrapper", default="False")
    parser.add_argument("--sensors", default="")
    parser.add_argument("--render_mode", default="rgb_array")  # "human" or "rgb_array".
    # NOTE: to get (nicer) 'human' rendering to work, you need to fix the compatibility issue between mujoco>3.0 and gymnasium: https://github.com/Farama-Foundation/Gymnasium/issues/749
    args = parser.parse_args()

    kwargs = vars(args).copy()
    kwargs.pop("env")
    kwargs.pop("render_mode")
    kwargs.pop("high_level_policy_path") # added this line to replace random sampling with high-level policy
    if kwargs["keyframe"] is None:
        kwargs.pop("keyframe")
    print(f"arguments: {kwargs}")

    # Test offscreen rendering
    print(f"Test offscreen mode...")
    env = gym.make(args.env, render_mode="rgb_array", **kwargs)
    ob, _ = env.reset()
    if isinstance(ob, dict):
        print(f"ob_space = {env.observation_space}")
        print(f"ob = ")
        for k, v in ob.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
    print(f"ac_space = {env.action_space.shape}")

    img = env.render()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_env_img.png", rgb_img)

    # Test online rendering with interactive viewer
    print(f"Test onscreen mode...")
    env = gym.make(args.env, render_mode=args.render_mode, **kwargs)
    ob, _ = env.reset()
    

    # Load model in two steps using the separated loader
    agent = get_agent(args.policy_path, args.mean_path, args.var_path, args.policy_type, args.env)
    agent = load_checkpoint(agent, args.high_level_policy_path)


    # load high-level policy
    if isinstance(ob, dict):
        print(f"ob_space = {env.observation_space}")
        print(f"ob = ")
        for k, v in ob.items():
            print(f"  {k}: {v.shape}")
            assert (
                v.shape == env.observation_space.spaces[k].shape
            ), f"{v.shape} != {env.observation_space.spaces[k].shape}"
        assert ob.keys() == env.observation_space.spaces.keys()
    else:
        print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
        assert env.observation_space.shape == ob.shape
    print(f"ac_space = {env.action_space.shape}")
    # print("observation:", ob)
    env.render()
    ret = 0
    step=0
    while True:

        # action = env.action_space.sample()

        # Get action from TD-MPC2 agent
        if isinstance(ob, dict):
            # Handle dictionary observations
            ob_tensor = torch.cat([torch.FloatTensor(v.flatten()) for v in ob.values()])
        else:
            ob_tensor = torch.FloatTensor(ob)
        
        ob_tensor = ob_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        with torch.no_grad():
            action = agent.act(ob_tensor, t0=step==0, eval_mode=True)
            if isinstance(action, torch.Tensor):
                action = action.squeeze().numpy()

        ob, rew, terminated, truncated, info = env.step(action)
        img = env.render()
        ret += rew
        step += 1

        if args.render_mode == "rgb_array":
            cv2.imshow("test_env", img[:, :, ::-1])
            cv2.waitKey(1)

        if terminated or truncated:
            ret = 0
            step = 0
            env.reset()
    env.close()
