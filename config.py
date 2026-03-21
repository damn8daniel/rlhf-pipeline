"""
Configuration for the RLHF pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0
    eos_token_id: int = 2


@dataclass
class SFTConfig:
    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    log_interval: int = 50
    save_path: str = "checkpoints/sft_model.pt"


@dataclass
class RewardConfig:
    epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    margin: float = 0.5
    log_interval: int = 50
    save_path: str = "checkpoints/reward_model.pt"


@dataclass
class PPOConfig:
    # PPO hyperparameters
    epochs: int = 100
    batch_size: int = 8
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    lr_policy: float = 1e-5
    lr_value: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # GAE
    gamma: float = 1.0
    lam: float = 0.95

    # PPO clipping
    clip_eps: float = 0.2
    clip_value: float = 0.2

    # Loss coefficients
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    kl_target: Optional[float] = 6.0

    # Adaptive KL
    adaptive_kl: bool = True
    kl_horizon: float = 10000.0

    # Generation
    max_gen_len: int = 64
    temperature: float = 0.8
    top_k: int = 50

    # Logging / saving
    log_interval: int = 10
    save_interval: int = 50
    save_path: str = "checkpoints/ppo_model.pt"
    reward_model_path: str = "checkpoints/reward_model.pt"
    sft_model_path: str = "checkpoints/sft_model.pt"
