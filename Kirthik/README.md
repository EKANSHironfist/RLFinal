# PPO 

This project implements the Proximal Policy Optimization (PPO) reinforcement learning (RL) algorithm using PyTorch and OpenAI Gym environments.
Please make sure all the necessary code files are in the same directory so the program runs correctly.
## 📁 Project Structure

```
.
├── ppo.py           # Proximal Policy Optimization (PPO)
├── networks.py      # Shared policy and value network definitions
├── plotting.py      # Plotting utility for visualizing training progress
├── main.py          # Runs the algorithm
├── README.md        # Project documentation
```

## Implemented Algorithm

- **PPO**: Proximal Policy Optimization using clipped surrogate objectives.

The algorithm is implemented modularly and rely on shared network architectures from `networks.py`.
Graph of the algorithm is plotted using plot_learning_curves function in `plotting.py`

##  How to Run

###  Run  Algorithm

To run the algorithm use:

```bash
python main.py
```

This script trains the PPO algorithm and saves reward statistics for visualization.

## 📦 Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```
