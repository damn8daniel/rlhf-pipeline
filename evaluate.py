"""
Evaluation utilities for the RLHF pipeline.

- Reward distribution analysis (before/after alignment)
- KL divergence tracking between policy and reference
- Generation quality comparison
"""
import os
import argparse
from typing import List, Dict, Optional
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, PPOConfig
from models.policy import PolicyModel, ReferenceModel
from models.reward_model import RewardModel
from data.prompt_dataset import PromptDataset, prompt_collate_fn
from utils import set_seed, get_device, get_logger, load_checkpoint

logger = get_logger("Evaluate")


@torch.no_grad()
def compute_reward_distribution(
    policy: PolicyModel,
    reward_model: RewardModel,
    prompt_loader: DataLoader,
    device: torch.device,
    max_gen_len: int = 64,
    n_batches: int = 10,
) -> Dict[str, float]:
    """
    Generate responses and compute reward distribution statistics.

    Returns:
        Dict with mean, std, min, max, median of rewards.
    """
    policy.eval()
    reward_model.eval()
    all_rewards = []

    for i, batch in enumerate(prompt_loader):
        if i >= n_batches:
            break

        prompt_ids = batch["prompt_ids"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)

        # Generate
        full_seq, _ = policy.generate(
            prompt_ids, max_new_tokens=max_gen_len, temperature=0.8, top_k=50
        )

        # Compute attention mask for full sequence
        attn_mask = (full_seq != policy.pad_token_id).long()

        # Score with reward model
        rewards = reward_model(full_seq, attn_mask)
        all_rewards.append(rewards.cpu())

    all_rewards = torch.cat(all_rewards)

    stats = {
        "reward/mean": all_rewards.mean().item(),
        "reward/std": all_rewards.std().item(),
        "reward/min": all_rewards.min().item(),
        "reward/max": all_rewards.max().item(),
        "reward/median": all_rewards.median().item(),
        "n_samples": len(all_rewards),
    }
    return stats


@torch.no_grad()
def compute_kl_divergence(
    policy: PolicyModel,
    ref_model: ReferenceModel,
    prompt_loader: DataLoader,
    device: torch.device,
    max_gen_len: int = 64,
    n_batches: int = 10,
) -> Dict[str, float]:
    """
    Estimate KL(policy || reference) by generating from the policy
    and comparing log-probabilities.

    Returns:
        Dict with mean and max per-token KL.
    """
    policy.eval()
    all_kl = []

    for i, batch in enumerate(prompt_loader):
        if i >= n_batches:
            break

        prompt_ids = batch["prompt_ids"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)
        prompt_len = prompt_ids.shape[1]

        # Generate from policy
        full_seq, _ = policy.generate(
            prompt_ids, max_new_tokens=max_gen_len, temperature=0.8, top_k=50
        )
        response_ids = full_seq[:, prompt_len:]
        response_len = response_ids.shape[1]

        if response_len == 0:
            continue

        attn_mask = (full_seq != policy.pad_token_id).long()

        # Policy log-probs
        logits_p, _ = policy(full_seq, attn_mask)
        resp_logits_p = logits_p[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
        logprobs_p = F.log_softmax(resp_logits_p, dim=-1)
        token_lp_p = logprobs_p.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

        # Reference log-probs
        logits_r, _ = ref_model.forward(full_seq, attn_mask)
        resp_logits_r = logits_r[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
        logprobs_r = F.log_softmax(resp_logits_r, dim=-1)
        token_lp_r = logprobs_r.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

        # KL per token
        kl = ReferenceModel.compute_kl_penalty(token_lp_p, token_lp_r)

        # Mask padding
        resp_mask = (response_ids != policy.pad_token_id).float()
        kl = kl * resp_mask

        all_kl.append(kl.sum(dim=1) / resp_mask.sum(dim=1).clamp(min=1))

    if not all_kl:
        return {"kl/mean": 0.0, "kl/max": 0.0}

    all_kl = torch.cat(all_kl)
    return {
        "kl/mean": all_kl.mean().item(),
        "kl/max": all_kl.max().item(),
        "kl/std": all_kl.std().item(),
    }


@torch.no_grad()
def compare_generations(
    sft_model: PolicyModel,
    rlhf_model: PolicyModel,
    reward_model: RewardModel,
    prompt_loader: DataLoader,
    device: torch.device,
    max_gen_len: int = 64,
    n_samples: int = 5,
) -> List[Dict]:
    """
    Generate responses from both the SFT and RLHF models on the same prompts
    and compare their reward scores.

    Returns:
        List of dicts with prompt, sft_response, rlhf_response, and their rewards.
    """
    sft_model.eval()
    rlhf_model.eval()
    reward_model.eval()

    comparisons = []
    total = 0

    for batch in prompt_loader:
        if total >= n_samples:
            break

        prompt_ids = batch["prompt_ids"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)
        B = prompt_ids.shape[0]

        # SFT generation
        sft_full, _ = sft_model.generate(
            prompt_ids, max_new_tokens=max_gen_len, temperature=0.8
        )
        sft_mask = (sft_full != sft_model.pad_token_id).long()
        sft_rewards = reward_model(sft_full, sft_mask)

        # RLHF generation
        rlhf_full, _ = rlhf_model.generate(
            prompt_ids, max_new_tokens=max_gen_len, temperature=0.8
        )
        rlhf_mask = (rlhf_full != rlhf_model.pad_token_id).long()
        rlhf_rewards = reward_model(rlhf_full, rlhf_mask)

        for j in range(min(B, n_samples - total)):
            comparisons.append({
                "prompt_tokens": prompt_ids[j].cpu().tolist(),
                "sft_tokens": sft_full[j].cpu().tolist(),
                "rlhf_tokens": rlhf_full[j].cpu().tolist(),
                "sft_reward": sft_rewards[j].item(),
                "rlhf_reward": rlhf_rewards[j].item(),
                "reward_improvement": (rlhf_rewards[j] - sft_rewards[j]).item(),
            })
            total += 1

    return comparisons


def full_evaluation(model_cfg: ModelConfig, ppo_cfg: PPOConfig) -> Dict:
    """Run full evaluation pipeline."""
    device = get_device()
    set_seed(42)

    # Load SFT model
    sft_model = PolicyModel(
        vocab_size=model_cfg.vocab_size,
        max_seq_len=model_cfg.max_seq_len,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
        pad_token_id=model_cfg.pad_token_id,
    ).to(device)

    if os.path.exists(ppo_cfg.sft_model_path):
        load_checkpoint(ppo_cfg.sft_model_path, sft_model, device=device)
        logger.info("Loaded SFT model")

    # Load RLHF model
    rlhf_model = PolicyModel(
        vocab_size=model_cfg.vocab_size,
        max_seq_len=model_cfg.max_seq_len,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
        pad_token_id=model_cfg.pad_token_id,
    ).to(device)

    if os.path.exists(ppo_cfg.save_path):
        load_checkpoint(ppo_cfg.save_path, rlhf_model, device=device)
        logger.info("Loaded RLHF model")

    # Reference model (frozen SFT)
    ref_model = ReferenceModel(sft_model).to(device)

    # Reward model
    reward_model = RewardModel(
        vocab_size=model_cfg.vocab_size,
        max_seq_len=model_cfg.max_seq_len,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
        pad_token_id=model_cfg.pad_token_id,
    ).to(device)

    if os.path.exists(ppo_cfg.reward_model_path):
        load_checkpoint(ppo_cfg.reward_model_path, reward_model, device=device)
        logger.info("Loaded reward model")

    reward_model.eval()

    # Prompt data
    prompt_dataset = PromptDataset(max_prompt_len=64, pad_token_id=model_cfg.pad_token_id)
    collate = partial(prompt_collate_fn, pad_token_id=model_cfg.pad_token_id)
    prompt_loader = DataLoader(
        prompt_dataset, batch_size=8, shuffle=False, collate_fn=collate
    )

    results = {}

    # 1. Reward distribution — SFT model
    logger.info("Computing reward distribution for SFT model...")
    sft_rewards = compute_reward_distribution(
        sft_model, reward_model, prompt_loader, device, n_batches=5
    )
    results["sft_rewards"] = sft_rewards
    logger.info(f"SFT rewards: mean={sft_rewards['reward/mean']:.4f}, "
                f"std={sft_rewards['reward/std']:.4f}")

    # 2. Reward distribution — RLHF model
    logger.info("Computing reward distribution for RLHF model...")
    rlhf_rewards = compute_reward_distribution(
        rlhf_model, reward_model, prompt_loader, device, n_batches=5
    )
    results["rlhf_rewards"] = rlhf_rewards
    logger.info(f"RLHF rewards: mean={rlhf_rewards['reward/mean']:.4f}, "
                f"std={rlhf_rewards['reward/std']:.4f}")

    # 3. KL divergence
    logger.info("Computing KL divergence (RLHF vs reference)...")
    kl_stats = compute_kl_divergence(
        rlhf_model, ref_model, prompt_loader, device, n_batches=5
    )
    results["kl"] = kl_stats
    logger.info(f"KL divergence: mean={kl_stats['kl/mean']:.4f}, "
                f"max={kl_stats['kl/max']:.4f}")

    # 4. Generation comparison
    logger.info("Comparing generations...")
    comparisons = compare_generations(
        sft_model, rlhf_model, reward_model, prompt_loader, device, n_samples=5
    )
    results["comparisons"] = comparisons
    for c in comparisons:
        logger.info(
            f"  SFT reward: {c['sft_reward']:.4f} | "
            f"RLHF reward: {c['rlhf_reward']:.4f} | "
            f"Improvement: {c['reward_improvement']:+.4f}"
        )

    # Summary
    reward_improvement = rlhf_rewards["reward/mean"] - sft_rewards["reward/mean"]
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info(f"  SFT reward (mean):    {sft_rewards['reward/mean']:.4f}")
    logger.info(f"  RLHF reward (mean):   {rlhf_rewards['reward/mean']:.4f}")
    logger.info(f"  Reward improvement:   {reward_improvement:+.4f}")
    logger.info(f"  KL divergence (mean): {kl_stats['kl/mean']:.4f}")
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLHF Pipeline")
    parser.add_argument("--sft_model_path", type=str, default="checkpoints/sft_model.pt")
    parser.add_argument("--rlhf_model_path", type=str, default="checkpoints/ppo_model.pt")
    parser.add_argument("--reward_model_path", type=str, default="checkpoints/reward_model.pt")
    args = parser.parse_args()

    model_cfg = ModelConfig()
    ppo_cfg = PPOConfig()
    ppo_cfg.sft_model_path = args.sft_model_path
    ppo_cfg.save_path = args.rlhf_model_path
    ppo_cfg.reward_model_path = args.reward_model_path

    full_evaluation(model_cfg, ppo_cfg)


if __name__ == "__main__":
    main()
