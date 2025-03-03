import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any

def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    将批次数据移动到指定设备
    
    Args:
        batch: 批次数据字典
        device: 目标设备
        
    Returns:
        移动到目标设备的批次数据
    """
    return {k: v.to(device) for k, v in batch.items()}

def create_masks(instruction_count: torch.Tensor, max_instr_count: int, x: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    创建指令级别的注意力掩码
    
    Args:
        instruction_count: 每个样本的指令数量 [batch_size]
        max_instr_count: 最大指令数量
        x: 输入张量，形状为 [batch_size, max_instr_count, max_instr_length]
        
    Returns:
        指令级别的掩码 [batch_size, max_instr_count]
    """
    batch_size = instruction_count.size(0)
    
    # 生成指令级别掩码（屏蔽填充指令）
    # [batch_size, max_instr_count]，True表示需要屏蔽
    instr_mask = torch.arange(max_instr_count, device=instruction_count.device)[None, :] >= instruction_count[:, None]
    
    return instr_mask

def pad_sequences(sequences: List[List[int]], max_len: Optional[int] = None, pad_value: int = 0) -> np.ndarray:
    """
    填充序列到相同长度
    
    Args:
        sequences: 整数序列列表
        max_len: 最大长度，如果为None则使用最长序列的长度
        pad_value: 填充值
        
    Returns:
        填充后的序列数组
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded_seqs = np.full((len(sequences), max_len), pad_value, dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq)] = seq
    
    return padded_seqs

def batch_sequences(sequences: List[List[int]], batch_size: int, shuffle: bool = False, pad_value: int = 0) -> List[np.ndarray]:
    """
    将序列分批
    
    Args:
        sequences: 整数序列列表
        batch_size: 批次大小
        shuffle: 是否打乱顺序
        pad_value: 填充值
        
    Returns:
        批次列表
    """
    indices = list(range(len(sequences)))
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_seqs = [sequences[idx] for idx in batch_indices]
        
        # 找出此批次中最长序列的长度
        max_len = max(len(seq) for seq in batch_seqs)
        
        # 填充序列
        padded_batch = pad_sequences(batch_seqs, max_len, pad_value)
        
        batches.append(padded_batch)
    
    return batches
