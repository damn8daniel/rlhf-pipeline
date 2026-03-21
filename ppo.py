"""
PPO (Proximal Policy Optimization) trainer for RLHF.

Implements:
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function loss with optional clipping
- Entropy bonus
- KL penalty from the reference model
- Mini-batch updates
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PPOConfig
from models.policy import PolicyModel, ReferenceModel
from models.reward_model import RewardModel
from utils import whiten, get_logger

logger = get_logger("PPO")


@dataclass
class RolloutBatch:
    """Stores a batch of rollout data for PPO training."""
    queries: torch.Tensor         # (B, prompt_len)
    responses: torch.Tensor       # (B, response_len)
    sequences: torch.Tensor       # (B, prompt_len + response_len)
    attention_mask: torch.Tensor  # (B, total_len)
    logprobs: torch.Tensor        # (B, response_len)
    ref_logprobs: torch.Tensor    # (B, response_len)
    values: torch.Tensor          # (B, response_len)
    rewards: torch.Tensor         # (B, response_len) — reward placed at last token
    advantages: torch.Tensor      # (B, response_len)
    returns: torch.Tensor         # (B, response_len)


class PPOTrainer:
    """
    PPO trainer that coordinates:
    1. Rollout generation (policy generates responses to prompts)
    2. Reward scoring (reward model scores responses)
    3. Advantage estimation via GAE
    4. PPO parameter updates with clipped objective
    """

    def __init__(
        self,
        policy: PolicyModel,
        ref_model: ReferenceModel,
        reward_model: RewardModel,
        config: PPOConfig,
        device: torch.device,
    ):
        self.policy = policy
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config
        self.device = device

        # Separate optimizer groups — policy and value head may use different LRs
        policy_params = []
        value_params = []
        for name, param in self.policy.named_parameters():
            if "value_head" in name:
                value_params.append(param)
            else:
                policy_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": policy_params, "lr": config.lr_policy},
                {"params": value_params, "lr": config.lr_value},
            ],
            weight_decay=config.weight_decay,
        )

        # Adaptive KL coefficient
        self.kl_coef = config.kl_coef

    @torch.no_grad()
    def generate_rollouts(
        self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor
    ) -> RolloutBatch:
        """
        Generate responses from the policy, compute log-probs, values,
        reference log-probs, and rewards.

        Args:
            prompt_ids: (B, prompt_len) padded prompt tokens
            prompt_mask: (B, prompt_len) attention mask for prompts
        """
        self.policy.eval()
        B = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]

        # 1. Generate responses
        full_seq, gen_logprobs = self.policy.generate(
            prompt_ids,
            max_new_tokens=self.config.max_gen_len,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            eos_token_id=2,
        )
        response_ids = full_seq[:, prompt_len:]
        response_len = response_ids.shape[1]

        if response_len == 0:
            # Edge case: nothing generated
            response_ids = torch.full((B, 1), 2, device=self.device, dtype=torch.long)
            response_len = 1
            gen_logprobs = torch.zeros(B, 1, device=self.device)

        # Truncate gen_logprobs to match response length
        gen_logprobs = gen_logprobs[:, :response_len]

        # Build full sequence and mask
        sequences = torch.cat([prompt_ids, response_ids], dim=1)
        total_len = sequences.shape[1]
        response_mask = (response_ids != self.policy.pad_token_id).long()
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

        # 2. Compute policy log-probs and values for the full sequence
        self.policy.eval()
        logits, values = self.policy(sequences, attention_mask)

        # Extract response-portion log-probs
        # logits at position t predicts token at position t+1
        response_logits = logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
        response_log_probs = F.log_softmax(response_logits, dim=-1)
        # Gather log-probs for the actual generated tokens
        token_logprobs = response_log_probs.gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Values for response tokens
        response_values = values[:, prompt_len - 1 : prompt_len - 1 + response_len]

        # 3. Reference model log-probs
        ref_logits, _ = self.ref_model.forward(sequences, attention_mask)
        ref_response_logits = ref_logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
        ref_log_probs = F.log_softmax(ref_response_logits, dim=-1)
        ref_token_logprobs = ref_log_probs.gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1)

        # 4. Compute rewards from the reward model
        reward_scores = self.reward_model(sequences, attention_mask)  # (B,)

        # Place scalar reward at the last response token, zeros elsewhere
        rewards = torch.zeros(B, response_len, device=self.device)
        for i in range(B):
            last_idx = response_mask[i].sum().long() - 1
            last_idx = max(last_idx, 0)
            rewards[i, last_idx] = reward_scores[i]

        # 5. Subtract KL penalty from rewards (per-token)
        kl_penalty = ReferenceModel.compute_kl_penalty(token_logprobs, ref_token_logprobs)
        rewards_with_kl = rewards - self.kl_coef * kl_penalty

        # 6. Compute advantages and returns via GAE
        advantages, returns = self._compute_gae(
            rewards_with_kl, response_values, response_mask
        )

        return RolloutBatch(
            queries=prompt_ids,
            responses=response_ids,
            sequences=sequences,
            attention_mask=attention_mask,
            logprobs=token_logprobs,
            ref_logprobs=ref_token_logprobs,
            values=response_values.detach(),
            rewards=rewards_with_kl,
            advantages=advantages,
            returns=returns,
        )

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE-lambda).

        A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Args:
            rewards: (B, T) per-token rewards (with KL penalty baked in)
            values: (B, T) value estimates
            mask: (B, T) binary mask for valid tokens

        Returns:
            advantages: (B, T) GAE advantages
            returns: (B, T) discounted returns (advantages + values)
        """
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=rewards.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(B, device=rewards.device)
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.config.gamma * next_value - values[:, t]
            last_gae = delta + self.config.gamma * self.config.lam * last_gae
            last_gae = last_gae * mask[:, t]
            advantages[:, t] = last_gae

        returns = advantages + values
        return advantages, returns

    def train_step(self, rollout: RolloutBatch) -> Dict[str, float]:
        """
        Perform PPO updates on a rollout batch.
        Splits the batch into mini-batches and runs multiple PPO epochs.

        Returns:
            Dictionary of training statistics.
        """
        self.policy.train()

        B, response_len = rollout.responses.shape
        prompt_len = rollout.queries.shape[1]

        # Normalize advantages
        advantages = whiten(rollout.advantages)

        stats = {
            "loss/policy": 0.0,
            "loss/value": 0.0,
            "loss/entropy": 0.0,
            "loss/total": 0.0,
            "policy/approx_kl": 0.0,
            "policy/clip_fraction": 0.0,
            "returns/mean": rollout.returns.mean().item(),
            "rewards/mean": rollout.rewards.mean().item(),
        }
        n_updates = 0

        indices = list(range(B))
        for ppo_epoch in range(self.config.ppo_epochs):
            # Shuffle
            import random
            random.shuffle(indices)

            for start in range(0, B, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, B)
                mb_idx = indices[start:end]
                mb_size = len(mb_idx)

                # Slice mini-batch
                mb_sequences = rollout.sequences[mb_idx]
                mb_attention_mask = rollout.attention_mask[mb_idx]
                mb_responses = rollout.responses[mb_idx]
                mb_old_logprobs = rollout.logprobs[mb_idx]
                mb_old_values = rollout.values[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = rollout.returns[mb_idx]

                # Forward pass
                logits, values = self.policy(mb_sequences, mb_attention_mask)

                # Extract response portion
                resp_logits = logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
                resp_values = values[:, prompt_len - 1 : prompt_len - 1 + response_len]

                # New log-probs
                new_log_probs = F.log_softmax(resp_logits, dim=-1)
                new_token_logprobs = new_log_probs.gather(
                    2, mb_responses.unsqueeze(-1)
                ).squeeze(-1)

                # Entropy
                probs = F.softmax(resp_logits, dim=-1)
                entropy = -(probs * new_log_probs).sum(-1).mean()

                # Ratio and clipped objective
                log_ratio = new_token_logprobs - mb_old_logprobs
                ratio = torch.exp(log_ratio)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with optional clipping
                if self.config.clip_value > 0:
                    values_clipped = mb_old_values + torch.clamp(
                        resp_values - mb_old_values,
                        -self.config.clip_value,
                        self.config.clip_value,
                    )
                    vf_loss1 = (resp_values - mb_returns) ** 2
                    vf_loss2 = (values_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * ((resp_values - mb_returns) ** 2).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # Statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.config.clip_eps)
                        .float()
                        .mean()
                        .item()
                    )

                stats["loss/policy"] += policy_loss.item()
                stats["loss/value"] += value_loss.item()
                stats["loss/entropy"] += entropy.item()
                stats["loss/total"] += loss.item()
                stats["policy/approx_kl"] += approx_kl
                stats["policy/clip_fraction"] += clip_frac
                n_updates += 1

        # Average stats
        for k in stats:
            if k.startswith("loss/") or k.startswith("policy/"):
                stats[k] /= max(n_updates, 1)

        # Adaptive KL coefficient
        if self.config.adaptive_kl and self.config.kl_target is not None:
            mean_kl = stats["policy/approx_kl"]
            if mean_kl > 1.5 * self.config.kl_target:
                self.kl_coef *= 1.5
            elif mean_kl < self.config.kl_target / 1.5:
                self.kl_coef /= 1.5
            self.kl_coef = max(self.kl_coef, 0.001)
            stats["policy/kl_coef"] = self.kl_coef

        return stats
