# Utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np


def _is_step_tuple(run):
    """Heuristic: (step, reward) pairs instead of plain reward list."""
    return isinstance(run[0], (tuple, list, np.ndarray)) and len(run[0]) == 2


def plot_learning_curves(results: dict[str, list], fname: str):
    """
    results – {algo_name: [ run1 , run2 , … ]}
        runᵢ can be
            • [reward, reward, …]                     (episodes on x-axis)
            • [(step, reward), (step, reward), …]     (steps on x-axis)
    """
    plt.figure(figsize=(9, 5))
    for algo, runs in results.items():
        for i, run in enumerate(runs):
            label = algo if i == 0 else None           # only one legend entry / algo
            if _is_step_tuple(run):
                steps, rews = zip(*run)
                plt.plot(steps, rews, label=label, alpha=0.7)
            else:
                plt.plot(range(len(run)), run, label=label, alpha=0.7)

    plt.xlabel("Environment steps")
    plt.ylabel("Total reward")
    plt.title("Learning Curves Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_comparison_boxplot(results: dict[str, list], fname: str):
    """Unchanged – still compares final reward of each run."""
    finals = {algo: [run[-1] if not _is_step_tuple(run) else run[-1][1]
                     for run in runs]
              for algo, runs in results.items()}
    plt.figure()
    plt.boxplot(finals.values(), labels=finals.keys())
    plt.ylabel("Final episode reward")
    plt.title("Final-Return Comparison")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
