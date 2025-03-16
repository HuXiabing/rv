from .tokenizer import RISCVTokenizer
import torch
from .gnn_dataset import *
from .dataset import *
from torch_geometric.loader import DataLoader as PyGDataLoader

__all__ = ["get_dataloader",
           "RISCVGraphDataset"]

def get_dataloader(model_type, dataset_path: str,
                  batch_size: int = 32,
                  shuffle: bool = True,
                  num_workers: int = 4) -> torch.utils.data.DataLoader:
    """
    Create a data loader

    Args:
        dataset_path: Path to the dataset json file
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of data loading threads

    Returns:
        Data loader
    """

    if model_type.lower() == "gnn":
        # dataset = RISCVDataset(dataset_path)
        # return torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=batch_size,
        #     shuffle=shuffle,
        #     num_workers=num_workers,
        #     pin_memory=True)
        # dataset = RISCVGraphDataset(dataset_path, cache_dir="./cache", rebuild_cache=False)
        # return PyGDataLoader(
        #         dataset,
        #         batch_size=batch_size,
        #         shuffle=shuffle,
        #         num_workers=num_workers,
        #         follow_batch=['x']
        #     )
        dataset = RISCVGraphDataset(dataset_path, cache_dir="./cache", rebuild_cache=False)
        return PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate
        )

    elif model_type.lower() == "transformer":

        dataset = DatasetWithDistanceWeight(dataset_path)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True)

    elif model_type.lower() == "lstm":

        dataset = RNNDataset(dataset_path)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True)

    else:
        raise ValueError(f"Model type {model_type} not supported")




