import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

class BaseModel(nn.Module):
    """所有模型的基类"""
    
    def __init__(self, config):
        """
        初始化基础模型
        
        Args:
            config: 配置对象
        """
        super(BaseModel, self).__init__()
        self.config = config
        self.device = torch.device(config.device)
    
    def forward(self, x, instruction_count=None):
        """
        前向传播方法，子类必须实现
        
        Args:
            x: 输入数据 [batch_size, max_instr_count, max_instr_length]
            instruction_count: 每个样本的指令数量 [batch_size]
            
        Returns:
            模型输出
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    def save(self, path: str) -> None:
        """
        保存模型状态和配置
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {
            'model_state': self.state_dict(),
            'config': self.config.__dict__,
            'model_type': self.__class__.__name__
        }
        
        torch.save(state_dict, path)
        print(f"模型已保存到 {path}")
    
    @classmethod
    def load(cls, path: str, config=None, device=None) -> 'BaseModel':
        """
        加载模型
        
        Args:
            path: 模型路径
            config: 配置对象（如果为None，则从checkpoint加载）
            device: 加载模型的设备（如果为None，则使用配置中的设备）
            
        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # 如果没有提供config，则从checkpoint加载
        if config is None:
            from config.config import Config
            config = Config(**checkpoint['config'])
        
        # 确定设备
        if device is not None:
            config.device = device
        
        # 根据模型类型创建实例
        from models.model_registry import get_model
        model = get_model(config)
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state'])
        model.to(torch.device(config.device))
        
        print(f"已从 {path} 加载模型")
        return model
    
    def count_parameters(self) -> int:
        """
        计算模型可训练参数的数量
        
        Returns:
            参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self) -> None:
        """冻结所有模型参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """解冻所有模型参数"""
        for param in self.parameters():
            param.requires_grad = True
