<p align="center">
  <h1 align="center">RLHF Pipeline</h1>
  <p align="center">
    Reinforcement Learning from Human Feedback &mdash; a complete three-stage alignment pipeline for language models, implemented from scratch in PyTorch.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/from-scratch-orange.svg" alt="From Scratch">
</p>

---

## Highlights

- **Full 3-stage RLHF** -- SFT, reward model training, and PPO alignment in one codebase
- **PPO from scratch** -- GAE advantage estimation, clipped surrogate objective, value function loss, entropy bonus
- **Reward model** -- transformer backbone with scalar head, trained with Bradley-Terry pairwise ranking
- **Adaptive KL penalty** -- controllable divergence from the reference (SFT) policy
- **GPT-style policy** -- causal transformer with full training infrastructure
- **Evaluation suite** -- reward distribution analysis, KL tracking, generation quality comparison
- **Configurable** -- all hyperparameters exposed via dataclasses in `config.py`

---

## Architecture

```
Stage 1: SFT            Stage 2: Reward Model      Stage 3: PPO
+----------------+      +---------------------+     +---------------------+
| Supervised     |      | (prompt, chosen,    |     | Policy (active)     |
| Fine-Tuning    |----->|  rejected) pairs    |---->| Reference (frozen)  |
| on demos       |      | Bradley-Terry loss  |     | PPO + KL penalty    |
+----------------+      +---------------------+     +---------------------+
     Base LM                  Preference Data             Aligned LM
```

### Stage 1: Supervised Fine-Tuning (SFT)

Fine-tune a base language model on high-quality demonstration data using standard cross-entropy (next-token prediction) loss. This produces the initial policy and the frozen reference model for Stage 3.

### Stage 2: Reward Model Training

Train a reward model on human preference data using the Bradley-Terry pairwise framework:

$$P(y_w \succ y_l \mid x) = \sigma\bigl(r(x, y_w) - r(x, y_l)\bigr)$$

The reward model shares the transformer backbone but replaces the language head with a scalar reward head. Trained with margin ranking loss on `(prompt, chosen, rejected)` triplets.

### Stage 3: PPO with KL Penalty

Optimize the policy to maximize reward while staying close to the reference model:

$$\max_\pi\ \mathbb{E}_{x \sim D,\, y \sim \pi}\bigl[r(x, y)\bigr] - \beta \cdot \text{KL}\bigl[\pi \;\|\; \pi_{\text{ref}}\bigr]$$

PPO implementation includes:
- **Generalized Advantage Estimation (GAE)** with configurable $\gamma$ and $\lambda$
- **Clipped surrogate objective** ($\epsilon = 0.2$)
- **Value function clipping** for stable learning
- **Entropy bonus** to encourage exploration
- **Adaptive KL coefficient** -- $\beta$ adjusts automatically to maintain a target KL

---

## Quick Start

### Installation

```bash
git clone https://github.com/damn8daniel/rlhf-pipeline.git
cd rlhf-pipeline
pip install -r requirements.txt
```

### Stage 1: Supervised Fine-Tuning

```bash
python sft.py \
    --data data/sft_train.jsonl \
    --epochs 3 \
    --batch_size 16 \
    --lr 1e-4
```

### Stage 2: Train the Reward Model

```bash
python reward_training.py \
    --data data/preferences.jsonl \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-5
```

Data format (JSON lines):
```json
{"prompt": "What is...", "chosen": "What is... The answer is...", "rejected": "What is... I don't know"}
```

### Stage 3: PPO Alignment

```bash
python rl_training.py \
    --sft_model_path checkpoints/sft_model.pt \
    --reward_model_path checkpoints/reward_model.pt \
    --epochs 100 \
    --kl_coef 0.1
```

### Evaluate

```bash
python evaluate.py \
    --policy_checkpoint checkpoints/ppo_model.pt \
    --reward_model_path checkpoints/reward_model.pt \
    --num_samples 200
```

---

## Configuration

All hyperparameters are organized in `config.py` as dataclasses:

### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Transformer hidden dimension |
| `n_heads` | 4 | Number of attention heads |
| `n_layers` | 4 | Number of transformer layers |
| `d_ff` | 512 | Feed-forward hidden dimension |
| `max_seq_len` | 256 | Maximum sequence length |
| `vocab_size` | 10000 | Vocabulary size |

### PPO

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_eps` | 0.2 | PPO clipping epsilon |
| `gamma` | 1.0 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `kl_coef` | 0.1 | KL penalty coefficient |
| `kl_target` | 6.0 | Target KL for adaptive coefficient |
| `ppo_epochs` | 4 | PPO update epochs per batch |
| `temperature` | 0.8 | Sampling temperature during rollouts |
| `top_k` | 50 | Top-k filtering during generation |

---

## Project Structure

```
rlhf-pipeline/
├── models/
│   ├── policy.py              # Policy and reference models, KL computation
│   └── reward_model.py        # Reward model with Bradley-Terry loss
├── data/
│   ├── preference_dataset.py  # (prompt, chosen, rejected) triplets
│   └── prompt_dataset.py      # Prompts for RL rollouts
├── ppo.py                     # PPO trainer (GAE, clipping, KL penalty, mini-batches)
├── sft.py                     # Stage 1: supervised fine-tuning
├── reward_training.py         # Stage 2: reward model training
├── rl_training.py             # Stage 3: PPO training with reward model
├── evaluate.py                # Reward distribution, KL tracking, generation analysis
├── config.py                  # All hyperparameters (ModelConfig, SFTConfig, RewardConfig, PPOConfig)
├── utils.py                   # Utilities (seeding, logging, scheduling, checkpointing)
└── requirements.txt
```

---

## Key Implementation Details

- **Rollout buffer** -- stores queries, responses, log-probs, ref-log-probs, values, rewards, advantages, and returns for PPO updates
- **Advantage whitening** -- advantages are normalized (zero mean, unit variance) before computing the policy loss
- **Adaptive KL** -- the KL coefficient $\beta$ increases when KL exceeds the target and decreases when it is below, maintaining stable alignment
- **Mini-batch PPO** -- rollout data is split into mini-batches for multiple update epochs per batch of experience
- **Separate LR** -- policy and value heads use different learning rates (`1e-5` and `1e-4` by default)

---

## Tech Stack

- **PyTorch** >= 2.0 -- model and training
- **Transformers** (HuggingFace) -- tokenizer utilities
- **NumPy** -- data processing
- **tqdm** -- progress bars

---

## References

- Ouyang, L., et al. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, 2022)
- Stiennon, N., et al. [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) (2020)
- Schulman, J., et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017)
- Ziegler, D., et al. [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) (2019)

---

## License

MIT
