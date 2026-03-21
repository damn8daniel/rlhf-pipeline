"""
Reward model: transformer backbone with a scalar reward head.
Trained with pairwise ranking loss (Bradley-Terry model).
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        if attention_mask is not None:
            attn = attn.masked_fill(
                attention_mask[:, None, None, :] == 0, float("-inf")
            )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class RewardModel(nn.Module):
    """
    Reward model that assigns a scalar reward to a (prompt, response) sequence.
    Uses the hidden state at the last non-padding token as the sequence
    representation, then projects to a scalar through a reward head.

    Training uses the Bradley-Terry pairwise ranking loss:
        loss = -log(sigmoid(r_chosen - r_rejected))
    optionally with a margin:
        loss = -log(sigmoid(r_chosen - r_rejected - margin))
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Scalar reward head
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scalar reward for each sequence in the batch.

        Args:
            input_ids: (B, T) token ids
            attention_mask: (B, T) binary mask (1 = attend, 0 = ignore)

        Returns:
            rewards: (B,) scalar reward per sequence
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)  # (B, T, d_model)

        # Use the last non-padding token's hidden state
        if attention_mask is not None:
            # Find index of last non-padding token per sequence
            seq_lengths = attention_mask.sum(dim=1).long() - 1  # (B,)
            seq_lengths = seq_lengths.clamp(min=0)
            last_hidden = x[torch.arange(B, device=x.device), seq_lengths]
        else:
            last_hidden = x[:, -1, :]

        rewards = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return rewards

    @staticmethod
    def ranking_loss(
        rewards_chosen: torch.Tensor,
        rewards_rejected: torch.Tensor,
        margin: float = 0.0,
    ) -> torch.Tensor:
        """
        Bradley-Terry pairwise ranking loss with optional margin.

        P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        loss = -log P(chosen > rejected)
             = -log sigmoid(r_chosen - r_rejected - margin)

        Args:
            rewards_chosen: (B,) rewards for preferred completions
            rewards_rejected: (B,) rewards for rejected completions
            margin: minimum desired margin between chosen and rejected

        Returns:
            Scalar loss value
        """
        return -F.logsigmoid(rewards_chosen - rewards_rejected - margin).mean()

    @staticmethod
    def margin_ranking_loss(
        rewards_chosen: torch.Tensor,
        rewards_rejected: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """
        Margin-based ranking loss:
            loss = max(0, margin - (r_chosen - r_rejected))

        Args:
            rewards_chosen: (B,) rewards for preferred completions
            rewards_rejected: (B,) rewards for rejected completions
            margin: desired margin

        Returns:
            Scalar loss value
        """
        return F.margin_ranking_loss(
            rewards_chosen,
            rewards_rejected,
            target=torch.ones_like(rewards_chosen),
            margin=margin,
        )

    @staticmethod
    def accuracy(
        rewards_chosen: torch.Tensor,
        rewards_rejected: torch.Tensor,
    ) -> float:
        """Fraction of pairs where chosen has higher reward than rejected."""
        return (rewards_chosen > rewards_rejected).float().mean().item()
