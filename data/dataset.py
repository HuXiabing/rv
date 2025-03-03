import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, Union, Optional, Tuple

class RISCVDataset(Dataset):
    """RISC-V指令数据集"""
    
    def __init__(self, h5_path: str):
        """
        初始化数据集
        
        Args:
            h5_path: HDF5文件路径
        """
        self.h5_path = h5_path
        
        # 读取数据集大小
        with h5py.File(h5_path, 'r') as f:
            self.num_samples = f.attrs['num_samples']
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples
    
    # def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    #     """
    #     获取数据集中的一个样本
    #
    #     Args:
    #         idx: 样本索引
    #
    #     Returns:
    #         包含特征和标签的字典
    #     """
    #     # 打开HDF5文件
    #     with h5py.File(self.h5_path, 'r') as f:
    #         # 获取特征和标签
    #         X = torch.tensor(f['X'][idx], dtype=torch.long)
    #         instruction_count = torch.tensor(f['instruction_counts'][idx], dtype=torch.long)
    #         Y = torch.tensor(f['Y'][idx], dtype=torch.float)
    #
    #     return {
    #         'X': X,
    #         'instruction_count': instruction_count,
    #         'Y': Y
    #     }
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据集中的一个样本

        Args:
            idx: 样本索引

        Returns:
            包含特征和标签的字典
        """
        # 获取实际索引
        actual_idx = self.indices[idx] if hasattr(self, 'indices') else idx

        # 打开HDF5文件
        with h5py.File(self.h5_path, 'r') as f:
            # 获取特征和标签
            X = torch.tensor(f['X'][actual_idx], dtype=torch.long)
            instruction_count = torch.tensor(f['instruction_counts'][actual_idx], dtype=torch.long)
            Y = torch.tensor(f['Y'][actual_idx], dtype=torch.float)

            # 尝试获取原始指令文本
            instruction_text = None
            if 'instruction_text' in f:
                instruction_text = f['instruction_text'][actual_idx]

        sample = {
            'X': X,
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
    创建数据加载器
    
    Args:
        dataset_path: 数据集HDF5文件路径
        batch_size: 批量大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        
    Returns:
        数据加载器
    """
    dataset = RISCVDataset(dataset_path)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
