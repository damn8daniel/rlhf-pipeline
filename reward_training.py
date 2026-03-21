"""
Stage 2: Reward Model Training.

Train a reward model on human preference data using
Bradley-Terry pairwise ranking loss.
"""
import os
import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import ModelConfig, RewardConfig
from models.reward_model import RewardModel
from data.preference_dataset import PreferenceDataset, preference_collate_fn
from utils import (
    set_seed,
    get_device,
    get_logger,
    save_checkpoint,
    count_parameters,
    get_cosine_schedule_with_warmup,
)

logger = get_logger("RewardTraining")


def train_reward_model(
    model_cfg: ModelConfig, reward_cfg: RewardConfig
) -> RewardModel:
    """Train the reward model on preference data."""
    device = get_device()
    set_seed(42)

    logger.info("Initializing reward model...")
    model = RewardModel(
        vocab_size=model_cfg.vocab_size,
        max_seq_len=model_cfg.max_seq_len,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
        pad_token_id=model_cfg.pad_token_id,
    ).to(device)
    logger.info(f"Reward model parameters: {count_parameters(model):,}")

    # Dataset
    dataset = PreferenceDataset(
        max_seq_len=model_cfg.max_seq_len, pad_token_id=model_cfg.pad_token_id
    )

    # Train/val split
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    collate = partial(preference_collate_fn, pad_token_id=model_cfg.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=reward_cfg.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=reward_cfg.batch_size, shuffle=False, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=reward_cfg.lr, weight_decay=reward_cfg.weight_decay
    )
    total_steps = len(train_loader) * reward_cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, reward_cfg.warmup_steps, total_steps
    )

    global_step = 0
    best_val_acc = 0.0

    for epoch in range(reward_cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        for batch_idx, batch in enumerate(train_loader):
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            r_chosen = model(chosen_ids, chosen_mask)
            r_rejected = model(rejected_ids, rejected_mask)

            # Bradley-Terry ranking loss
            loss = RewardModel.ranking_loss(r_chosen, r_rejected, margin=reward_cfg.margin)
            acc = RewardModel.accuracy(r_chosen, r_rejected)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), reward_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            global_step += 1

            if global_step % reward_cfg.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_acc = epoch_acc / (batch_idx + 1)
                logger.info(
                    f"Epoch {epoch+1}/{reward_cfg.epochs} | "
                    f"Step {global_step} | "
                    f"Loss {avg_loss:.4f} | "
                    f"Acc {avg_acc:.4f}"
                )

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                chosen_ids = batch["chosen_ids"].to(device)
                chosen_mask = batch["chosen_mask"].to(device)
                rejected_ids = batch["rejected_ids"].to(device)
                rejected_mask = batch["rejected_mask"].to(device)

                r_chosen = model(chosen_ids, chosen_mask)
                r_rejected = model(rejected_ids, rejected_mask)

                loss = RewardModel.ranking_loss(r_chosen, r_rejected, margin=reward_cfg.margin)
                acc = RewardModel.accuracy(r_chosen, r_rejected)

                val_loss += loss.item()
                val_acc += acc
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        avg_val_acc = val_acc / max(val_batches, 1)
        logger.info(
            f"Epoch {epoch+1} | Val Loss {avg_val_loss:.4f} | Val Acc {avg_val_acc:.4f}"
        )

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_checkpoint(
                model, optimizer, epoch, global_step, reward_cfg.save_path,
                extra={"val_acc": best_val_acc},
            )
            logger.info(f"Best model saved (val_acc={best_val_acc:.4f})")

    logger.info(f"Reward model training complete. Best val accuracy: {best_val_acc:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Reward Model Training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    reward_cfg = RewardConfig()

    if args.epochs is not None:
        reward_cfg.epochs = args.epochs
    if args.batch_size is not None:
        reward_cfg.batch_size = args.batch_size
    if args.lr is not None:
        reward_cfg.lr = args.lr
    if args.margin is not None:
        reward_cfg.margin = args.margin

    train_reward_model(model_cfg, reward_cfg)


if __name__ == "__main__":
    main()
