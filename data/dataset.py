import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, Union, Optional, Tuple

class RISCVDataset(Dataset):

    def __init__(self, h5_path: str):

        self.h5_path = h5_path

        with h5py.File(h5_path, 'r') as f:
            self.num_samples = f.attrs['num_samples']
    
    def __len__(self) -> int:

        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a sample from the dataset

        Args:
            idx: Sample index

        Returns:
            Dictionary containing features and labels
        """

        actual_idx = self.indices[idx] if hasattr(self, 'indices') else idx

        with h5py.File(self.h5_path, 'r') as f:

            X = torch.tensor(f['X'][actual_idx], dtype=torch.long)
            instruction_count = torch.tensor(f['instruction_counts'][actual_idx], dtype=torch.long)
            Y = torch.tensor(f['Y'][actual_idx], dtype=torch.float)

            instruction_text = None
            if 'instruction_text' in f:
                instruction_text = f['instruction_text'][actual_idx]

        sample = {
            'X': X,  #encoded matrix
            'instruction_count': instruction_count,
            'Y': Y
        }

        if instruction_text is not None:
            sample['instruction_text'] = instruction_text

        return sample

def get_dataloader(dataset_path: str, 
                  batch_size: int = 32, 
                  shuffle: bool = True, 
                  num_workers: int = 4) -> torch.utils.data.DataLoader:
    """
    Create a data loader

    Args:
        dataset_path: Path to the dataset HDF5 file
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of data loading threads

    Returns:
        Data loader
    """
    dataset = RISCVDataset(dataset_path)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
