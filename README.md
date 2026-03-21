# RLHF Pipeline

Reinforcement Learning from Human Feedback — complete three-stage alignment pipeline for language models, implemented from scratch in PyTorch.

## Architecture

```
Stage 1: SFT          Stage 2: Reward Model     Stage 3: PPO
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ Supervised   │      │ (prompt,chosen,  │      │ Policy (active)  │
│ Fine-Tuning  │─────▶│  rejected) pairs │─────▶│ Reference (frozen)│
│ on demos     │      │ Bradley-Terry    │      │ PPO + KL penalty │
└─────────────┘      └──────────────────┘      └─────────────────┘
```

## Three Training Stages

### Stage 1: Supervised Fine-Tuning (SFT)
Fine-tune a pretrained language model on high-quality demonstration data using standard cross-entropy loss.

### Stage 2: Reward Model Training
Train a reward model on human preference data using the Bradley-Terry pairwise ranking framework:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

### Stage 3: PPO with KL Penalty
Optimize the policy using PPO with a KL divergence penalty from the reference model:

$$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi}[r(x, y)] - \beta \cdot KL[\pi || \pi_{ref}]$$

## Features

- **Reward Model**: Transformer backbone with scalar reward head, margin ranking loss
- **PPO Trainer**: GAE advantage estimation, clipped surrogate objective, value function loss, entropy bonus
- **KL Penalty**: Controllable divergence from reference policy
- **SFT**: Standard supervised fine-tuning with configurable hyperparameters

## Project Structure

```
rlhf-pipeline/
├── models/
│   ├── reward_model.py    # Reward model with Bradley-Terry loss
│   └── policy.py          # Policy and reference models, KL computation
├── data/
│   ├── preference_dataset.py  # (prompt, chosen, rejected) data
│   └── prompt_dataset.py      # Prompts for RL rollouts
├── ppo.py                 # PPO trainer with GAE, clipping, KL penalty
├── sft.py                 # Stage 1: supervised fine-tuning
├── reward_training.py     # Stage 2: reward model training
├── rl_training.py         # Stage 3: PPO training with reward
├── evaluate.py            # Reward distribution analysis
├── config.py              # All hyperparameters
└── utils.py               # Utilities
```

## Tech Stack

- PyTorch
- Transformers (HuggingFace)
- NumPy

## References

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT)
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) (Stiennon et al.)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al.)
