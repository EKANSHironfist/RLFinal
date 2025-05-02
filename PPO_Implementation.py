# Algorithms/PPO_Implementation.py
import torch, numpy as np, gymnasium as gym
from torch.distributions import Categorical
from torch.nn.functional import mse_loss
import torch.optim as optim
from Models.networks import PolicyNetwork, ValueNetwork


class PPO:
    def __init__(self,
                 env_name: str,
                 lr_policy=3e-4,
                 lr_value=1e-3,
                 gamma=0.99, lam=0.95,
                 clip_eps=0.2,
                 ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5,
                 n_envs=8, rollout_length=2048,
                 minibatch_size=64, epochs=10):
        # vector env ----------------------------------------------------------
        self.envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_name)
                                              for _ in range(n_envs)])
        obs_dim = self.envs.single_observation_space.shape[0]
        act_dim = self.envs.single_action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(obs_dim, act_dim).to(self.device)
        self.value  = ValueNetwork(obs_dim).to(self.device)

        for net in (self.policy, self.value):                          # orthogonal init
            for m in net.modules():
                if hasattr(m, "weight"):
                    torch.nn.init.orthogonal_(m.weight, gain=0.1)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

        self.opt_pi = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.opt_v  = optim.Adam(self.value .parameters(), lr=lr_value)

        # hyper-params --------------------------------------------------------
        self.gamma, self.lam      = gamma, lam
        self.clip_eps             = clip_eps
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm        = max_grad_norm
        self.n_envs, self.rollout_length = n_envs, rollout_length
        self.minibatch_size, self.epochs = minibatch_size, epochs

    # --------------------------------------------------------------------- utils
    def normalize_obs(self, obs: np.ndarray):
        if not hasattr(self, "_obs_mean"):
            self._obs_mean = np.zeros_like(obs[0], np.float32)
            self._obs_var  = np.ones_like(obs[0],  np.float32)
            self._obs_cnt  = 1e-4
        batch_mean, batch_var = obs.mean(0), obs.var(0)
        batch_cnt  = len(obs)
        delta      = batch_mean - self._obs_mean
        tot_cnt    = self._obs_cnt + batch_cnt
        new_mean   = self._obs_mean + delta * batch_cnt / tot_cnt
        m_a        = self._obs_var * self._obs_cnt
        m_b        = batch_var     * batch_cnt
        M2         = m_a + m_b + delta ** 2 * self._obs_cnt * batch_cnt / tot_cnt
        self._obs_mean, self._obs_var, self._obs_cnt = new_mean, M2 / tot_cnt, tot_cnt
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def compute_gae(self, rewards, masks, values):
        T, N = rewards.shape
        adv  = np.zeros_like(rewards, np.float32)
        last = 0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] * masks[t] - values[t]
            last  = delta + self.gamma * self.lam * masks[t] * last
            adv[t] = last
        return adv, adv + values[:-1]

    # --------------------------------------------------------------------- train
    def train(self, max_steps: int):
        """
        Returns: list[tuple[int, float]]
                [(step_of_episode_1, return_1), …]  – ready for step-based plots
        """
        step_ctr     = 0                     # global env steps across all workers
        ep_returns   = []                    # per-episode returns
        ep_steps     = []                    # env-step when episode finished
        current_rets = np.zeros(self.n_envs) # running return of each worker
        rollout_idx  = 0

        while step_ctr < max_steps:
            rollout_idx += 1

            # ------------- rollout collection
            mb_obs, mb_act, mb_logp, mb_rew, mb_mask, mb_val = [], [], [], [], [], []
            obs, _ = self.envs.reset()
            obs     = self.normalize_obs(obs)

            for _ in range(self.rollout_length):
                s_t  = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    logits = self.policy(s_t)
                    dist   = Categorical(logits)
                    value  = self.value(s_t).squeeze(-1)

                act   = dist.sample()
                log_p = dist.log_prob(act)

                nxt_obs, rew, term, trunc, _ = self.envs.step(act.cpu().numpy())
                done = term | trunc
                mask = 1.0 - done.astype(np.float32)

                # buffer -------------------------------------------------------
                mb_obs .append(obs)
                mb_act .append(act.cpu().numpy())
                mb_logp.append(log_p.cpu().numpy())
                mb_rew .append(rew)
                mb_mask.append(mask)
                mb_val .append(value.cpu().numpy())

                # per-episode bookkeeping -------------------------------------
                current_rets += rew
                for env_i, d in enumerate(done):
                    if d:
                        ep_returns.append(current_rets[env_i])
                        ep_steps  .append(step_ctr + self.n_envs)  # episode ends after step
                        current_rets[env_i] = 0.0

                obs = self.normalize_obs(nxt_obs)
                step_ctr += self.n_envs
                if step_ctr >= max_steps:
                    break

            # --------- bootstrap value of last state
            with torch.no_grad():
                last_val = self.value(torch.as_tensor(obs,
                                                     dtype=torch.float32,
                                                     device=self.device)).squeeze(-1).cpu().numpy()
            mb_val.append(last_val)

            # --------- prepare flat arrays
            T = len(mb_rew)
            obs_arr   = np.asarray(mb_obs   ).reshape(T * self.n_envs, -1)
            act_arr   = np.asarray(mb_act   ).reshape(T * self.n_envs)
            logp_arr  = np.asarray(mb_logp  ).reshape(T * self.n_envs)
            rew_arr   = np.asarray(mb_rew   ).reshape(T, self.n_envs)
            mask_arr  = np.asarray(mb_mask  ).reshape(T, self.n_envs)
            val_arr   = np.asarray(mb_val)              # (T+1, N)

            adv, ret = self.compute_gae(rew_arr, mask_arr, val_arr)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # --------- PPO update
            batch_size = T * self.n_envs
            inds = np.arange(batch_size)
            obs_t   = torch.as_tensor(obs_arr,  dtype=torch.float32, device=self.device)
            act_t   = torch.as_tensor(act_arr,  dtype=torch.long,   device=self.device)
            oldlp_t = torch.as_tensor(logp_arr, dtype=torch.float32, device=self.device)
            ret_t   = torch.as_tensor(ret.reshape(-1),  dtype=torch.float32, device=self.device)
            adv_t   = torch.as_tensor(adv.reshape(-1),  dtype=torch.float32, device=self.device)

            for _ in range(self.epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, self.minibatch_size):
                    mb = inds[start:start + self.minibatch_size]

                    dist = Categorical(self.policy(obs_t[mb]))
                    newlp = dist.log_prob(act_t[mb])
                    ratio = torch.exp(newlp - oldlp_t[mb])

                    surr1 = ratio * adv_t[mb]
                    surr2 = torch.clamp(ratio,
                                        1 - self.clip_eps,
                                        1 + self.clip_eps) * adv_t[mb]
                    loss_pi = -torch.min(surr1, surr2).mean() \
                              - self.ent_coef * dist.entropy().mean()

                    val_pred = self.value(obs_t[mb]).squeeze(-1)
                    loss_v   = self.vf_coef * mse_loss(val_pred, ret_t[mb])

                    self.opt_pi.zero_grad()
                    loss_pi.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.opt_pi.step()

                    self.opt_v.zero_grad()
                    loss_v.backward()
                    torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                    self.opt_v.step()

            # ------------------- progress print (every 10 finished episodes)
            if len(ep_returns) and len(ep_returns) % 10 == 0:
                avg = np.mean(ep_returns[-10:])
                print(f"Steps: {step_ctr:>7}, Episode: {len(ep_returns):>5}, "
                      f"Avg Reward: {avg:6.1f}")

        self.envs.close()
        return list(zip(ep_steps, ep_returns))
