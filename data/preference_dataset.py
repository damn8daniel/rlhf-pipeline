"""
Preference dataset for reward model training.
Each sample is a triplet: (prompt, chosen_response, rejected_response).
"""
import json
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    """
    Dataset of human preference comparisons.

    Each item contains:
        - prompt_ids: tokenized prompt
        - chosen_ids: tokenized preferred (prompt + chosen) completion
        - rejected_ids: tokenized rejected (prompt + rejected) completion

    Data format (JSON lines):
        {"prompt": [1, 5, 3, ...], "chosen": [1, 5, 3, 7, 8, ...], "rejected": [1, 5, 3, 4, 2, ...]}

    Or raw text with a tokenizer:
        {"prompt": "What is...", "chosen": "What is... The answer is...", "rejected": "What is... I don't know"}
    """

    def __init__(
        self,
        data: Optional[List[Dict]] = None,
        file_path: Optional[str] = None,
        max_seq_len: int = 256,
        pad_token_id: int = 0,
    ):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        if data is not None:
            self.data = data
        elif file_path is not None:
            self.data = []
            with open(file_path, "r") as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        else:
            # Generate synthetic preference data for demonstration
            self.data = self._generate_synthetic_data()

    def _generate_synthetic_data(self, n_samples: int = 500) -> List[Dict]:
        """Generate synthetic preference data for testing."""
        import random
        random.seed(42)
        data = []
        for _ in range(n_samples):
            prompt_len = random.randint(5, 20)
            chosen_len = random.randint(10, 40)
            rejected_len = random.randint(10, 40)
            prompt = [random.randint(3, 9999) for _ in range(prompt_len)]
            chosen = prompt + [random.randint(3, 9999) for _ in range(chosen_len)]
            rejected = prompt + [random.randint(3, 9999) for _ in range(rejected_len)]
            data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        chosen_ids = item["chosen"][: self.max_seq_len]
        rejected_ids = item["rejected"][: self.max_seq_len]

        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
        }


def preference_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Collate preference pairs into padded batches.

    Returns:
        chosen_ids: (B, T_chosen) padded chosen token ids
        chosen_mask: (B, T_chosen) attention mask
        rejected_ids: (B, T_rejected) padded rejected token ids
        rejected_mask: (B, T_rejected) attention mask
    """
    chosen_ids = [item["chosen_ids"] for item in batch]
    rejected_ids = [item["rejected_ids"] for item in batch]

    chosen_ids = _pad_sequences(chosen_ids, pad_token_id)
    rejected_ids = _pad_sequences(rejected_ids, pad_token_id)

    chosen_mask = (chosen_ids != pad_token_id).long()
    rejected_mask = (rejected_ids != pad_token_id).long()

    return {
        "chosen_ids": chosen_ids,
        "chosen_mask": chosen_mask,
        "rejected_ids": rejected_ids,
        "rejected_mask": rejected_mask,
    }


def _pad_sequences(
    sequences: List[torch.Tensor], pad_value: int
) -> torch.Tensor:
    """Pad a list of variable-length tensors to the same length."""
    max_len = max(seq.size(0) for seq in sequences)
    padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : seq.size(0)] = seq
    return padded
