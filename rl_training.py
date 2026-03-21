"""
Stage 3: RL Training with PPO.

Use the trained reward model to optimize the policy via PPO
with KL penalty against the reference (SFT) model.
"""
import os
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, PPOConfig
from models.policy import PolicyModel, ReferenceModel
from models.reward_model import RewardModel
from data.prompt_dataset import PromptDataset, prompt_collate_fn
from ppo import PPOTrainer
from utils import (
    set_seed,
    get_device,
    get_logger,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
)

logger = get_logger("RLTraining")


def train_rl(model_cfg: ModelConfig, ppo_cfg: PPOConfig) -> PolicyModel:
    """Run PPO training loop."""
    device = get_device()
    set_seed(42)

    # 1. Load SFT model as policy
    logger.info("Loading SFT model as initial policy...")
    policy = PolicyModel(
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
        load_checkpoint(ppo_cfg.sft_model_path, policy, device=device)
        logger.info(f"Loaded SFT checkpoint from {ppo_cfg.sft_model_path}")
    else:
        logger.warning("No SFT checkpoint found. Using randomly initialized policy.")

    logger.info(f"Policy parameters: {count_parameters(policy):,}")

    # 2. Create frozen reference model
    logger.info("Creating frozen reference model...")
    ref_model = ReferenceModel(policy).to(device)

    # 3. Load reward model
    logger.info("Loading reward model...")
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
        logger.info(f"Loaded reward model from {ppo_cfg.reward_model_path}")
    else:
        logger.warning("No reward model checkpoint found. Using random reward model.")

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    # 4. Create prompt dataset and dataloader
    prompt_dataset = PromptDataset(
        max_prompt_len=64, pad_token_id=model_cfg.pad_token_id
    )
    collate = partial(prompt_collate_fn, pad_token_id=model_cfg.pad_token_id)
    prompt_loader = DataLoader(
        prompt_dataset,
        batch_size=ppo_cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    # 5. Initialize PPO trainer
    trainer = PPOTrainer(
        policy=policy,
        ref_model=ref_model,
        reward_model=reward_model,
        config=ppo_cfg,
        device=device,
    )

    # 6. Training loop
    logger.info("Starting PPO training...")
    global_step = 0
    all_stats = []

    for epoch in range(ppo_cfg.epochs):
        for batch_idx, batch in enumerate(prompt_loader):
            prompt_ids = batch["prompt_ids"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)

            # Generate rollouts
            rollout = trainer.generate_rollouts(prompt_ids, prompt_mask)

            # PPO update
            stats = trainer.train_step(rollout)
            stats["epoch"] = epoch
            stats["step"] = global_step
            all_stats.append(stats)

            global_step += 1

            if global_step % ppo_cfg.log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{ppo_cfg.epochs} | "
                    f"Step {global_step} | "
                    f"Reward {stats['rewards/mean']:.4f} | "
                    f"Policy Loss {stats['loss/policy']:.4f} | "
                    f"Value Loss {stats['loss/value']:.4f} | "
                    f"KL {stats['policy/approx_kl']:.4f} | "
                    f"Clip {stats['policy/clip_fraction']:.4f}"
                )

            if global_step % ppo_cfg.save_interval == 0:
                save_checkpoint(
                    policy,
                    trainer.optimizer,
                    epoch,
                    global_step,
                    ppo_cfg.save_path,
                    extra={"kl_coef": trainer.kl_coef},
                )
                logger.info(f"Checkpoint saved at step {global_step}")

    # Final save
    save_checkpoint(
        policy,
        trainer.optimizer,
        ppo_cfg.epochs,
        global_step,
        ppo_cfg.save_path,
        extra={"kl_coef": trainer.kl_coef},
    )
    logger.info(f"PPO training complete. Model saved to {ppo_cfg.save_path}")
    return policy


def main():
    parser = argparse.ArgumentParser(description="Stage 3: PPO Training with Reward Model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr_policy", type=float, default=None)
    parser.add_argument("--kl_coef", type=float, default=None)
    parser.add_argument("--sft_model_path", type=str, default=None)
    parser.add_argument("--reward_model_path", type=str, default=None)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    ppo_cfg = PPOConfig()

    if args.epochs is not None:
        ppo_cfg.epochs = args.epochs
    if args.batch_size is not None:
        ppo_cfg.batch_size = args.batch_size
    if args.lr_policy is not None:
        ppo_cfg.lr_policy = args.lr_policy
    if args.kl_coef is not None:
        ppo_cfg.kl_coef = args.kl_coef
    if args.sft_model_path is not None:
        ppo_cfg.sft_model_path = args.sft_model_path
    if args.reward_model_path is not None:
        ppo_cfg.reward_model_path = args.reward_model_path

    train_rl(model_cfg, ppo_cfg)


if __name__ == "__main__":
    main()
