"""
Hidden-state trajectory dataset for CRF training.

Loads pre-collected teacher trajectories (h*, Δ*) from disk
and pairs them with tokenized (x, y*) for the SFT component.
"""

import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional
from pathlib import Path


class TrajectoryDataset(Dataset):
    """Dataset of teacher hidden-state trajectories.

    Each sample contains:
      - input_ids, attention_mask, labels: for SFT loss
      - h_stars: list of L tensors (seq_len, d) — teacher hidden states
      - delta_stars: list of L tensors (seq_len, d) — teacher residual fluxes
    """

    def __init__(
        self,
        trajectory_dir: str,
        tokenizer_name: str,
        max_seq_len: int = 4096,
    ):
        self.trajectory_dir = Path(trajectory_dir)
        self.files = sorted(self.trajectory_dir.glob("trajectory_*.pt"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len

        if not self.files:
            raise ValueError(f"No trajectory files found in {trajectory_dir}")
        print(f"Loaded {len(self.files)} trajectory files from {trajectory_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")

        # Tokenized sequence for SFT
        tokens = data["tokens"]
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]

        input_ids = tokens[:-1]
        labels = tokens[1:]
        attention_mask = torch.ones_like(input_ids)

        # Teacher hidden states (truncate to match token length)
        T = len(input_ids)
        h_stars = [h[:T] for h in data["h_stars"]]
        delta_stars = [d[:T] for d in data["delta_stars"]]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "h_stars": h_stars,
            "delta_stars": delta_stars,
        }
