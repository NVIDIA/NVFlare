from typing import Tuple

import pandas as pd
import torch
from et_task_processor import ETTaskProcessor
from torch.utils.data import Dataset


class XORDataset(Dataset):
    """XOR Dataset following PyTorch conventions."""

    def __init__(self, csv_path: str):
        """Initialize XOR dataset.

        Args:
            csv_path: Path to CSV file containing XOR data
        """
        # Load data from CSV
        df = pd.DataFrame(pd.read_csv(csv_path))

        self.X = torch.tensor(df[["x1", "x2"]].values, dtype=torch.float32)
        self.Y = torch.tensor(df["y"].values, dtype=torch.int64)

        self._verify_data()

    def _verify_data(self) -> None:
        """Verify data format and shapes."""
        if not isinstance(self.X, torch.Tensor) or not isinstance(self.Y, torch.Tensor):
            raise TypeError("X and Y must be torch.Tensor")

        if self.X.shape[1] != 2:
            raise ValueError(f"Expected X to have 2 features, got {self.X.shape[1]}")

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must have same number of samples")

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            tuple: (x, y) pair of input and target
        """
        return self.X[idx], self.Y[idx]


class XORETTaskProcessor(ETTaskProcessor):
    """Task processor for XOR dataset."""

    def get_dataset(self, data_path: str) -> Dataset:
        return XORDataset(data_path)
