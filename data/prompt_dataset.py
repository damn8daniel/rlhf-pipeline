"""
Prompt dataset for RL rollout generation (PPO stage).
"""
import json
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """
    Dataset of prompts for RL rollout generation.

    Each item is a tokenized prompt that the policy will complete.

    Data format (JSON lines):
        {"prompt": [1, 5, 3, ...]}
    """

    def __init__(
        self,
        data: Optional[List[Dict]] = None,
        file_path: Optional[str] = None,
        max_prompt_len: int = 64,
        pad_token_id: int = 0,
    ):
        self.max_prompt_len = max_prompt_len
        self.pad_token_id = pad_token_id

        if data is not None:
            self.data = data
        elif file_path is not None:
            self.data = []
            with open(file_path, "r") as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        else:
            self.data = self._generate_synthetic_prompts()

    def _generate_synthetic_prompts(self, n_samples: int = 200) -> List[Dict]:
        """Generate synthetic prompts for testing."""
        import random
        random.seed(42)
        data = []
        for _ in range(n_samples):
            prompt_len = random.randint(5, 30)
            prompt = [random.randint(3, 9999) for _ in range(prompt_len)]
            data.append({"prompt": prompt})
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.data[idx]["prompt"][: self.max_prompt_len]
        return {"prompt_ids": torch.tensor(prompt, dtype=torch.long)}


def prompt_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Collate prompts into a padded batch. Left-pads prompts so that
    generation can proceed from the right edge.

    Returns:
        prompt_ids: (B, T) left-padded prompt token ids
        prompt_mask: (B, T) attention mask
    """
    prompts = [item["prompt_ids"] for item in batch]
    max_len = max(p.size(0) for p in prompts)

    # Left-pad for causal generation
    padded = torch.full((len(prompts), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(prompts), max_len, dtype=torch.long)

    for i, p in enumerate(prompts):
        offset = max_len - p.size(0)
        padded[i, offset:] = p
        mask[i, offset:] = 1

    return {
        "prompt_ids": padded,
        "prompt_mask": mask,
    }
