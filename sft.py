"""
Stage 1: Supervised Fine-Tuning (SFT).

Train the base language model on high-quality demonstration data
using standard cross-entropy (next-token prediction) loss.
"""
import os
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import ModelConfig, SFTConfig
from models.policy import PolicyModel
from utils import (
    set_seed,
    get_device,
    get_logger,
    save_checkpoint,
    count_parameters,
    get_cosine_schedule_with_warmup,
)

logger = get_logger("SFT")


class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    Each sample is a tokenized sequence (prompt + response) for causal LM training.
    """

    def __init__(
        self,
        data: List[Dict] = None,
        max_seq_len: int = 256,
        pad_token_id: int = 0,
        n_samples: int = 1000,
    ):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        if data is not None:
            self.data = data
        else:
            # Generate synthetic SFT data
            self.data = self._generate_synthetic(n_samples)

    def _generate_synthetic(self, n: int) -> List[Dict]:
        import random
        random.seed(42)
        data = []
        for _ in range(n):
            seq_len = random.randint(20, self.max_seq_len - 1)
            tokens = [random.randint(3, 9999) for _ in range(seq_len)]
            tokens.append(2)  # EOS
            data.append({"input_ids": tokens})
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]["input_ids"][: self.max_seq_len]
        return {"input_ids": torch.tensor(tokens, dtype=torch.long)}


def sft_collate_fn(batch, pad_token_id=0):
    seqs = [item["input_ids"] for item in batch]
    max_len = max(s.size(0) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(seqs), max_len), -100, dtype=torch.long)

    for i, s in enumerate(seqs):
        input_ids[i, : s.size(0)] = s
        labels[i, : s.size(0)] = s

    attention_mask = (input_ids != pad_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train_sft(model_cfg: ModelConfig, sft_cfg: SFTConfig) -> PolicyModel:
    """Run supervised fine-tuning."""
    device = get_device()
    set_seed(42)

    logger.info("Initializing policy model for SFT...")
    model = PolicyModel(
        vocab_size=model_cfg.vocab_size,
        max_seq_len=model_cfg.max_seq_len,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
        pad_token_id=model_cfg.pad_token_id,
    ).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    dataset = SFTDataset(max_seq_len=model_cfg.max_seq_len, pad_token_id=model_cfg.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=sft_cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: sft_collate_fn(b, model_cfg.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=sft_cfg.lr, weight_decay=sft_cfg.weight_decay
    )
    total_steps = len(dataloader) * sft_cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, sft_cfg.warmup_steps, total_steps)

    global_step = 0
    model.train()

    for epoch in range(sft_cfg.epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, attention_mask)

            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % sft_cfg.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1}/{sft_cfg.epochs} | "
                    f"Step {global_step} | "
                    f"Loss {avg_loss:.4f} | "
                    f"LR {lr:.2e}"
                )

        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")

    save_checkpoint(model, optimizer, sft_cfg.epochs, global_step, sft_cfg.save_path)
    logger.info(f"SFT model saved to {sft_cfg.save_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Supervised Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    sft_cfg = SFTConfig()

    if args.epochs is not None:
        sft_cfg.epochs = args.epochs
    if args.batch_size is not None:
        sft_cfg.batch_size = args.batch_size
    if args.lr is not None:
        sft_cfg.lr = args.lr

    train_sft(model_cfg, sft_cfg)


if __name__ == "__main__":
    main()
