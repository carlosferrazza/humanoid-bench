import humanoid_bench
import gymnasium as gym
import numpy as np
import time
import cv2 
from time import sleep
from fast_td3 import ActorGNN, build_egnn_input
import humanoid_bench

def main():
    # Create environment
    env = gym.make(
        "h1-stand-v0",
        render_mode="rgb_array",
    )

    # Reset environment
    observation, info = env.reset()
    
    # Print environment info
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    try:
        # Main loop
        for episode in range(3):
            terminated = False
            truncated = False
            total_reward = 0
            steps = 0
            
            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)

                x, h, edge_index, edge_attr = build_egnn_input(
                    int((env.observation_space.shape[0] - 13) / 2),
                    env.unwrapped.named.data.qpos,
                    env.unwrapped.named.data.qvel,
                    env.unwrapped.named.data.xpos,
                )

                print(x)
                print(h)
                print(edge_index)
                print(edge_attr)

                break

                # frame = env.render()
                # cv2.imshow('Humanoid Simulation', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                #     break
                
                # total_reward += reward
                # steps += 1
                # time.sleep(0.01)
            
            print(f"Episode {episode + 1} finished after {steps} steps. Total reward: {total_reward:.2f}")
            observation, info = env.reset()
    
    finally:
        cv2.destroyAllWindows()  # Clean up OpenCV windows
        env.close()

if __name__ == "__main__":
    main()