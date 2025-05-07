import gymnasium as gym
import torch
import random
import numpy as np
import argparse
import os
from ppo import PPO
from plotting import plot_learning_curves

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def run_ppo(num_runs=5, max_steps=1000000, seed=42):
    results = []
    
    for run in range(num_runs):
        print(f"\nRunning PPO, Run {run+1}/{num_runs}")
        set_seeds(seed + run)
        env = gym.make("CartPole-v1")
        agent = PPO(env)
        step_rewards = agent.train(max_steps=max_steps)
        results.append(step_rewards)
        env.close()
    
    # Save Results
    os.makedirs('results', exist_ok=True)
    import pickle
    with open("results/PPO_rewards.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Training')
    parser.add_argument("--runs", type=int, default=5, help="Number of runs for PPO")
    parser.add_argument("--steps", type=int, default=1000000, help="Number of environment steps per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    results = run_ppo(args.runs, args.steps, args.seed)
    plot_learning_curves({"PPO": results}, "ppo_learning_curve.png")