import torch
import torch.nn.functional as F
import numpy as np
from networks import PolicyNetwork, ValueNetwork
import torch.optim as optim


class PPO:
    def __init__(
        self,
        env,
        lr_policy: float = 5e-4,
        lr_value: float = 5e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        lambda_: float = 0.95,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        """Proximal Policy Optimisation agent.

        Args:
            env: Gymnasium‑style environment.
            lr_policy: Learning‑rate for the actor.
            lr_value: Learning‑rate for the critic.
            gamma: Reward discount factor.
            clip_eps: PPO clip parameter.
            lambda_: GAE decay parameter.
            batch_size: SGD mini‑batch size.
            entropy_coef: Coefficient for the entropy bonus (encourages exploration).
            max_grad_norm: Gradient‑norm clip threshold.
        """
        self.env = env

        self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.value_net = ValueNetwork(env.observation_space.shape[0])

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    # ---------------------------------------------------------------------
    #  Main interaction helpers
    # ---------------------------------------------------------------------

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy_net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    @staticmethod
    def _compute_gae(rewards, values, masks, next_value, gamma, lambda_):
        returns, advantages = [], []
        advantage = 0.0
        for t in reversed(range(len(rewards))):
            next_v = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_v * masks[t] - values[t]
            advantage = delta + gamma * lambda_ * advantage * masks[t]
            R = rewards[t] + gamma * (
                next_value if t == len(rewards) - 1 else returns[0] if returns else 0.0
            ) * masks[t]
            returns.insert(0, R)
            advantages.insert(0, advantage)
        return torch.tensor(returns, dtype=torch.float32), torch.tensor(
            advantages, dtype=torch.float32
        )

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------

    def train(self, max_steps: int = 1_000_000):
        reward_records, step_rewards = [], []
        total_steps, episode = 0, 0

        while total_steps < max_steps:
            state, _ = self.env.reset()
            done, episode_reward = False, 0.0
            # Trajectory storages
            states, actions, rewards, masks, old_log_probs = [], [], [], [], []

            while not done and total_steps < max_steps:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                masks.append(1.0 - float(done))
                old_log_probs.append(log_prob)

                state = next_state
                episode_reward += reward
                total_steps += 1

                if total_steps % 1000 == 0:
                    avg_reward = np.mean(reward_records[-50:]) if reward_records else 0.0
                    step_rewards.append((total_steps, avg_reward))

            # ----------------------------------------------------------
            #  Compute returns & advantages (GAE)
            # ----------------------------------------------------------
            states_t = torch.FloatTensor(np.array(states))
            actions_t = torch.LongTensor(actions)
            old_log_probs_t = torch.FloatTensor(old_log_probs)
            values_t = self.value_net(states_t).squeeze(-1)
            with torch.no_grad():
                next_value = self.value_net(torch.FloatTensor(next_state).unsqueeze(0)).squeeze(-1)
            returns_t, advantages_t = self._compute_gae(
                rewards, values_t, masks, next_value, self.gamma, self.lambda_
            )
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            dataset = torch.utils.data.TensorDataset(
                states_t, actions_t, old_log_probs_t, returns_t, advantages_t
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )

            # ----------------------------------------------------------
            #  PPO updates (multiple epochs over the collected batch)
            # ----------------------------------------------------------
            for _ in range(5):
                for (
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_returns,
                    batch_advantages,
                ) in dataloader:
                    # Critic
                    values_pred = self.value_net(batch_states).squeeze(-1)
                    value_loss = F.mse_loss(values_pred, batch_returns)

                    # Actor
                    action_probs = self.policy_net(batch_states)
                    dist = torch.distributions.Categorical(action_probs)
                    curr_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()

                    ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                    policy_loss = -(
                        torch.min(surr1, surr2) + self.entropy_coef * entropy
                    ).mean()

                    # Back-prop -- Actor
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_net.parameters(), self.max_grad_norm
                    )
                    self.policy_optimizer.step()

                    # Back-prop -- Critic
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.value_net.parameters(), self.max_grad_norm
                    )
                    self.value_optimizer.step()

            # Book-keeping
            reward_records.append(episode_reward)
            episode += 1
            if episode % 10 == 0:
                avg_reward = np.mean(reward_records[-10:])
                print(
                    f"Steps: {total_steps}, Episode: {episode}, Avg Reward: {avg_reward:.1f}"
                )

        return step_rewards
