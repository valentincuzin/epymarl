# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_resigtry


class PPOLearnerV2:
    # inspired by this implementation https://github.com/marlbenchmark/on-policy
    # ADD:
    #   GAE compute return
    #   remove critic
    #   advantage norm
    #   clip value
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)

        critic_mask = mask.clone()

        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0

        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        for k in range(self.args.epochs):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            pi = mac_out
            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()
            # Calculate policy grad with mask

            pi[mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = (
                th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * advantages
            )

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            pg_loss = (
                -(
                    (th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask
                ).sum()
                / mask.sum()
            )

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "advantages_abs",
                "q_taken_mean",
                "gae_returns_means",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, batch, rewards, mask):
        # Optimise critic
        with th.no_grad():
            old_v = critic(batch).squeeze(3)
        v = critic(batch)[:, :-1].squeeze(3)

        # destandardise return
        if self.args.standardise_returns:
            old_v = old_v * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            v = v * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        gae_returns = self.gae_returns(
            rewards, mask, old_v, self.args.gae_lambda
            )

        if self.args.standardise_returns:
            self.ret_ms.update(gae_returns)
            gae_returns = (gae_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        advantages = gae_returns.detach() - v
        if self.args.normalise_advantages:
            advantages_copy = advantages.clone()[mask == 1]
            mean_advantages = th.mean(advantages_copy)
            std_advantages = th.std(advantages_copy)
            advantages = (advantages - mean_advantages) / std_advantages
            advantages[mask == 0] = 0.0
        
        if self.args.use_clipped_value_loss:           
            v_preds_clipped = old_v[:, :-1] + (v - old_v[:, :-1]).clamp(
                -self.args.eps_clip_v, self.args.eps_clip_v
            )

            loss_clipped = ((gae_returns.detach() - v_preds_clipped) ** 2)
            loss_original = ((gae_returns.detach() - v) ** 2)
            
            loss = th.max(loss_clipped, loss_original)
            loss = (loss * mask).sum() / mask.sum()
            
        else:
            loss = ((gae_returns.detach()-v)**2*mask).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "advantages_abs": [],
            "gae_returns_means": [],
            "q_taken_mean": [],
        }

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["advantages_abs"].append(
            (advantages.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["gae_returns_means"].append(
            (gae_returns * mask).sum().item() / mask_elems
        )

        return advantages, running_log

    def gae_returns(self, rewards, mask, values, gae_lambda=0.95):
        # inspired by this implementation https://github.com/marlbenchmark/on-policy
        mask = th.cat([mask, mask[:, -1:].clone()], dim=1)
        T = rewards.size(1)
        gae_values = th.zeros_like(values[:, :-1])  # (B, T)
        for t_start in range(T):
            gae_t = th.zeros_like(values[0, -1])
            for step in range(0, T - t_start):
                t = t_start + step
                # delta = r_t + gamma * V_{t+1} * mask_{t+1} - V_t
                delta = rewards[:, t] + self.args.gamma * values[:, t + 1] * mask[:, t + 1] - values[:, t]
                gae_t = delta + self.args.gamma * gae_lambda * mask[:, t + 1] * gae_t
            # return = advantage + V_t
            gae_values[:, t_start] = gae_t + values[:, t_start]
        return gae_values

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
