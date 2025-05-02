import gymnasium as gym
import torch
import random
import numpy as np
import argparse
import os
from Algorithms.reinforce import REINFORCE
from Algorithms.actor_critic import ActorCritic
from Algorithms.a2c import A2C
from Algorithms.dqn import DQNagent
from Algorithms.ppo import PPO as PPOStd
from Algorithms.PPO_Implementation import PPO as PPOImpl
from Utils.plotting import plot_learning_curves, plot_comparison_boxplot


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def run_algorithm(algo, num_runs=5, max_steps=200000, seed=42):
    # mapping CLI names to internal keys
    name_map = {
        'reinforce': 'REINFORCE',
        'actor_critic': 'ActorCritic',
        'a2c': 'A2C',
        'dqn': 'DQNAgent',
        'ppo': 'PPO',
        'ppo_implementation': 'PPO_IMPLEMENTATION'
    }
    algo_key = name_map.get(algo.lower())
    if not algo_key:
        raise ValueError(f"Unknown algorithm {algo}")

    class_map = {
        'REINFORCE': REINFORCE,
        'ActorCritic': ActorCritic,
        'A2C': A2C,
        'DQNAgent': DQNagent,
        'PPO': PPOStd,
        'PPO_IMPLEMENTATION': PPOImpl
    }

    results = []
    for run in range(num_runs):
        print(f"\nRunning {algo_key}, Run {run+1}/{num_runs}")
        set_seeds(seed + run)

        if algo_key in ('PPO', 'PPO_IMPLEMENTATION'):
            cls = class_map[algo_key]
            agent = cls(env_name='CartPole-v1')
            rewards = agent.train(max_steps)
        else:
            env = gym.make('CartPole-v1')
            cls = class_map[algo_key]
            agent = cls(env, learning_rate=0.0005, gamma=0.99) if algo_key.startswith(('REINFORCE','ActorCritic','A2C')) else cls(env)
            rewards = agent.train(max_steps=max_steps)
            env.close()

        results.append(rewards)

    os.makedirs('results', exist_ok=True)
    import pickle
    with open(f"results/{algo_key}_rewards.pkl", 'wb') as f:
        pickle.dump(results, f)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['reinforce','actor_critic','a2c','dqn','ppo','ppo_implementation','all'], default='all')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.algorithm == 'all':
        algos = ['reinforce','actor_critic','a2c','dqn','ppo','ppo_implementation']
        all_res = {algo: run_algorithm(algo, args.runs, args.steps, args.seed) for algo in algos}
        plot_learning_curves(all_res, 'all_algos.png')
        plot_comparison_boxplot(all_res, 'final_compare.png')
    else:
        res = run_algorithm(args.algorithm, args.runs, args.steps, args.seed)
        plot_learning_curves({args.algorithm: res}, f"{args.algorithm}_curve.png")
