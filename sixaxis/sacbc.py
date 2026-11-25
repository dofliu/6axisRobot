"""T204：SAC＋示範行為複製（Q-filter BC）——SB3 SAC 的子類別，只改寫 `train()` 的損失。

動機：只把示範放進回放緩衝（DDPGfD 式預填）時，示範轉移很快被自身經驗稀釋，且 actor 只透過尚未訓練好的 critic
間接受示範影響；T204 實測此法在 40k 步內成功率仍為 0（見報告）。因此加入兩個已發表的機制：

1. 示範專屬緩衝（Nair et al. 2018, "Overcoming Exploration in Reinforcement Learning with Demonstrations"）：
   每個梯度步另抽 n_demo 筆示範與 batch_size 筆自身經驗合併，critic 與 actor 都在合併批次上更新。
2. Q-filter 行為複製損失（同 Nair et al. 2018）＋TD3+BC 的尺度正規化（Fujimoto & Gu 2021,
   "A Minimalist Approach to Offline Reinforcement Learning"）：
     L_actor = λ·E[α_ent log π(a|s) − min_i Q_i(s, a)] + E_demo[ 1{Q(s, a_D) > Q(s, μ(s))}·‖μ(s) − a_D‖² ]
     λ = α_bc / mean|Q(s, a)|（detach），μ(s) = 確定性（tanh 均值）動作。
   Q-filter 只在示範動作的 Q 值優於目前策略時才模仿，策略超越示範後 BC 自動失效。
其餘（雙 Q、目標網路 Polyak、自動熵溫度）與 SB3 SAC 2.x `train()` 完全相同。
"""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class SACBC(SAC):
    alpha_bc: float = 2.5
    n_demo: int = 64

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "demo"]

    def set_demos(self, trans, alpha_bc: float = 2.5, n_demo: int = 64):
        """trans：[(o, a, r, o', terminated, truncated), ...]（`rlplace.collect_demos` 格式）。"""
        o, a, r, o2, term, _ = (np.array(x) for x in zip(*trans, strict=True))
        dev = self.device
        self.demo = {
            "obs": th.as_tensor(o, dtype=th.float32, device=dev),
            "act": th.as_tensor(a, dtype=th.float32, device=dev),
            "rew": th.as_tensor(r, dtype=th.float32, device=dev).reshape(-1, 1),
            "next": th.as_tensor(o2, dtype=th.float32, device=dev),
            "done": th.as_tensor(term, dtype=th.float32, device=dev).reshape(-1, 1),
        }
        self.alpha_bc, self.n_demo = alpha_bc, n_demo
        self.bc_stats = {"bc_loss": [], "q_filter": []}

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        demo = getattr(self, "demo", None)
        if demo is None:
            return super().train(gradient_steps, batch_size)
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        nd = len(demo["obs"])
        for gradient_step in range(gradient_steps):
            rd = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            idx = th.randint(0, nd, (self.n_demo,), device=self.device)
            obs = th.cat([rd.observations, demo["obs"][idx]])
            act = th.cat([rd.actions, demo["act"][idx]])
            rew = th.cat([rd.rewards, demo["rew"][idx]])
            nxt = th.cat([rd.next_observations, demo["next"][idx]])
            done = th.cat([rd.dones, demo["done"][idx]])
            discounts = self.gamma

            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.reshape(-1, 1)
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(nxt)
                next_q = th.cat(self.critic_target(nxt, next_actions), dim=1)
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = rew + (1 - done) * discounts * next_q
            current_q = self.critic(obs, act)
            critic_loss = 0.5 * sum(F.mse_loss(cq, target_q) for cq in current_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_pi = th.cat(self.critic(obs, actions_pi), dim=1)
            min_q_pi, _ = th.min(q_pi, dim=1, keepdim=True)
            lam = self.alpha_bc / min_q_pi.abs().mean().detach().clamp(min=1e-3)
            sac_loss = (ent_coef * log_prob - min_q_pi).mean()
            od, ad = demo["obs"][idx], demo["act"][idx]
            mu = self.actor(od, deterministic=True)
            with th.no_grad():
                q_d = th.min(th.cat(self.critic(od, ad), dim=1), dim=1, keepdim=True)[0]
                q_mu = th.min(th.cat(self.critic(od, mu), dim=1), dim=1, keepdim=True)[0]
                mask = (q_d > q_mu).float()
            bc_loss = (mask * ((mu - ad) ** 2).sum(dim=1, keepdim=True)).mean()
            actor_loss = lam * sac_loss + bc_loss
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            if self._n_updates % 1000 == 0:
                self.bc_stats["bc_loss"].append(float(bc_loss.item()))
                self.bc_stats["q_filter"].append(float(mask.mean().item()))
        self._n_updates += gradient_steps
