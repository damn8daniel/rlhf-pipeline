"""
Policy model (GPT-style causal language model) and frozen reference model
with KL divergence computation.
"""
import copy
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

        # Causal mask
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

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
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

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class PolicyModel(nn.Module):
    """
    GPT-style causal language model used as the policy in PPO.
    Outputs both token logits and a scalar value estimate per position.
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
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Value head (for PPO critic)
        self.value_head = nn.Linear(d_model, 1)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, T, vocab_size)
            values: (B, T)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        values = self.value_head(x).squeeze(-1)

        return logits, values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive generation. Returns generated token ids and their log-probs.
        """
        generated = input_ids.clone()
        all_logprobs = []

        for _ in range(max_new_tokens):
            if generated.shape[1] > self.max_seq_len:
                context = generated[:, -self.max_seq_len :]
            else:
                context = generated

            logits, _ = self.forward(context)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            log_probs = F.log_softmax(logits, dim=-1)
            token_logprob = log_probs.gather(1, next_token)
            all_logprobs.append(token_logprob)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        logprobs = torch.cat(all_logprobs, dim=1)
        return generated, logprobs


class ReferenceModel:
    """
    Frozen copy of the policy model used as the KL reference.
    Wraps a PolicyModel with no_grad and eval mode.
    """

    def __init__(self, policy_model: PolicyModel):
        self.model = copy.deepcopy(policy_model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.model(input_ids, attention_mask)

    def to(self, device: torch.device) -> "ReferenceModel":
        self.model.to(device)
        return self

    @staticmethod
    def compute_kl_penalty(
        logprobs_policy: torch.Tensor,
        logprobs_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token KL divergence: KL(policy || ref).
        Uses the approximation: KL = exp(log_ref - log_policy) - 1 - (log_ref - log_policy)
        which is the Schulman (2020) unbiased KL estimator: k3 approximation.

        Args:
            logprobs_policy: (B, T) log-probabilities under the policy
            logprobs_ref: (B, T) log-probabilities under the reference

        Returns:
            kl: (B, T) per-token KL divergence
        """
        log_ratio = logprobs_ref - logprobs_policy
        # k3 estimator: more numerically stable than naive KL
        kl = torch.exp(log_ratio) - 1.0 - log_ratio
        return kl
