import os
import json
import torch
from typing import Dict, Any, Optional

class Config:

    def __init__(
        self,
        model_type: str = "transformer",  # 'transformer', 'lstm', 'gnn'

        max_instr_length: int = 8,
        max_instr_count: int = 200,
        vocab_size: int = 2000,
        
        # model parameters
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        
        # training parameters
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 5,
        clip_grad_norm: float = 1.0,

        device: Optional[str] = None,
        
        # data paths
        raw_data_path: str = "data/labeled_data.json",
        processed_data_path: str = "data/processed_data.json",
        train_data_path: str = "data/train_data.h5",
        val_data_path: str = "data/val_data.h5",
        test_data_path: str = "data/test_data.h5",
        
        output_dir: str = "experiments",
        experiment_name: Optional[str] = None,
        
        seed: int = 42,
        verbose: bool = True,
        
        save_best_only: bool = True,
        save_freq: int = 1,
        max_checkpoints: int = 5,
        
        **kwargs
    ):
        self.model_type = model_type
        
        self.max_instr_length = max_instr_length
        self.max_instr_count = max_instr_count
        self.vocab_size = vocab_size
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.clip_grad_norm = clip_grad_norm
        
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
        # self.experiment_dir = os.path.join(output_dir, experiment_name)
        # self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        # self.log_dir = os.path.join(self.experiment_dir, "logs")
        # self.vocab_path = os.path.join(self.experiment_dir, "vocab.json")
        
        # 创建必要的目录
        # os.makedirs(self.checkpoint_dir, exist_ok=True)
        # os.makedirs(self.log_dir, exist_ok=True)
        
        self.seed = seed
        self.verbose = verbose
        
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()
    
    def update(self, **kwargs) -> None:
        """update config with new key-value pairs"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Config has no attribute {key}")
    
    def __str__(self) -> str:
        return f"Configuration for {self.model_type} model, experiment: {self.experiment_name}"
    
    def __repr__(self) -> str:
        attrs = []
        for key, value in sorted(self.__dict__.items()):
            attrs.append(f"{key}={value}")
        return f"Config({', '.join(attrs)})"
