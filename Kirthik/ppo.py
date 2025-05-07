import torch
import torch.nn.functional as F
import numpy as np
from networks import PolicyNetwork, ValueNetwork
import torch.optim as optim

class PPO:
    def __init__(self, env, lr_policy=0.0005, lr_value=0.0005, gamma=0.99, clip_eps=0.2, lambda_=0.95, batch_size=64):
        self.env = env
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.value_net = ValueNetwork(env.observation_space.shape[0])
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lambda_ = lambda_
        self.batch_size = batch_size
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy_net(state)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def compute_gae(self, rewards, values, masks, next_value):
        returns = []
        advantages = []
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            # At the last timestep, use next_value; otherwise, use values[t+1]
            next_v = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_v * masks[t] - values[t]
            advantage = delta + self.gamma * self.lambda_ * advantage * masks[t]
            R = rewards[t] + self.gamma * (next_value if t == len(rewards) - 1 else returns[0] if returns else 0) * masks[t]
            returns.insert(0, R)
            advantages.insert(0, advantage)
        
        return torch.tensor(returns).float(), torch.tensor(advantages).float()
    
    def train(self, max_steps=1000000):
        reward_records = []  # Store total reward per episode
        step_rewards = []  # Store (step, avg_reward) pairs
        total_steps = 0
        episode = 0

        while total_steps < max_steps:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            states, actions, rewards, masks, log_probs_old = [], [], [], [], []
            episode_steps = 0

            while not done:
                if total_steps >= max_steps:
                    break
                
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                masks.append(1 - float(done))
                log_probs_old.append(log_prob)

                state = next_state
                episode_reward += reward
                total_steps += 1
                episode_steps += 1
                
                if total_steps % 1000 == 0:
                    avg_reward = np.mean(reward_records[-50:]) if reward_records else 0
                    step_rewards.append((total_steps, avg_reward))
            
            if total_steps >= max_steps:
                break

            # Compute values and GAE
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            log_probs_old = torch.FloatTensor(log_probs_old)
            values = self.value_net(states).squeeze(-1)
            with torch.no_grad():
                next_value = self.value_net(torch.FloatTensor(next_state).unsqueeze(0)).squeeze(-1)
            returns, advantages = self.compute_gae(rewards, values, masks, next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Mini-batch updates
            dataset = torch.utils.data.TensorDataset(states, actions, log_probs_old, returns, advantages)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            for epoch in range(5):  # 5 epochs of updates
                for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                    values = self.value_net(batch_states).squeeze(-1)
                    action_probs = self.policy_net(batch_states)
                    dist = torch.distributions.Categorical(action_probs)
                    curr_log_probs = dist.log_prob(batch_actions)

                    ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages

                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(values, batch_returns)

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    self.value_optimizer.step()

            reward_records.append(episode_reward)
            episode += 1

            if episode % 10 == 0:
                avg_reward = np.mean(reward_records[-10:])
                print(f"Steps: {total_steps}, Episode: {episode}, Avg Reward: {avg_reward:.1f}")

        return step_rewards