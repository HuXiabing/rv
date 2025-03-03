import os
import torch
import numpy as np
import json
import h5py
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models import BaseModel
from data import RISCVDataProcessor, RISCVTokenizer


class RISCVPredictor:
    """RISC-V指令吞吐量预测器"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型检查点路径
            device: 推理设备，如果为None则自动选择
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 加载模型
        self.model = BaseModel.load(model_path, device=self.device)
        self.model.eval()
        
        # 获取模型配置
        checkpoint = torch.load(model_path, map_location="cpu")
        self.config = checkpoint.get("config", {})
        
        # 创建分词器和处理器
        from config.config import Config
        self.config_obj = Config(**self.config)
        self.processor = RISCVDataProcessor(self.config_obj)
        
        # 加载词汇表
        vocab_path = self.config_obj.vocab_path
        if os.path.exists(vocab_path):
            self.processor.tokenizer.load_vocab(vocab_path)
        else:
            raise ValueError(f"找不到词汇表文件: {vocab_path}")
        
        # 打印模型信息
        print(f"加载了 {self.config_obj.model_type.upper()} 模型，参数数量: {self.model.count_parameters():,}")
        print(f"推理设备: {self.device}")
    
    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        批量预测
        
        Args:
            batch: 输入批次，包含'X'和'instruction_count'
            
        Returns:
            预测吞吐量
        """
        with torch.no_grad():
            # 将数据移到设备
            x = batch['X'].to(self.device)
            instruction_count = batch['instruction_count'].to(self.device)
            
            # 前向传播
            output = self.model(x, instruction_count)
            
            return output
    
    def predict_from_hdf5(self, hdf5_path: str, batch_size: int = 32) -> Dict[str, Any]:
        """
        从HDF5文件进行预测
        
        Args:
            hdf5_path: HDF5文件路径
            batch_size: 批量大小
            
        Returns:
            预测结果字典
        """
        from data import get_dataloader
        
        # 创建数据加载器
        dataloader = get_dataloader(
            hdf5_path,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # 进行预测
        all_preds = []
        
        for batch in dataloader:
            # 预测当前批次
            output = self.predict_batch(batch)
            
            # 保存预测结果
            all_preds.extend(output.cpu().numpy())
        
        # 如果HDF5包含真实值，也加载它们
        with h5py.File(hdf5_path, 'r') as f:
            if 'Y' in f:
                y_true = f['Y'][:].tolist()
                result = {
                    "y_true": y_true,
                    "y_pred": all_preds
                }
            else:
                result = {
                    "y_pred": all_preds
                }
        
        return result
    
    def predict_from_json(self, json_data: Union[str, List[Dict[str, Any]]],
                         batch_size: int = 32) -> Dict[str, Any]:
        """
        从JSON数据进行预测
        
        Args:
            json_data: JSON文件路径或者数据列表
            batch_size: 批量大小
            
        Returns:
            预测结果字典
        """
        # 加载JSON数据
        if isinstance(json_data, str):
            with open(json_data, 'r') as f:
                input_data = json.load(f)
        else:
            input_data = json_data
        
        # 处理输入数据
        temp_hdf5 = "temp_inference.h5"
        self.processor.process_new_data(input_data, temp_hdf5, update_vocab=False)
        
        # 从HDF5进行预测
        result = self.predict_from_hdf5(temp_hdf5, batch_size)
        
        # 删除临时文件
        if os.path.exists(temp_hdf5):
            os.remove(temp_hdf5)
        
        # 检查输入数据是否包含真实值
        if "y_true" not in result and all("throughput" in item for item in input_data):
            result["y_true"] = [item["throughput"] for item in input_data]
        
        return result
    
    def predict_single(self, instructions: List[str]) -> float:
        """
        预测单个样本
        
        Args:
            instructions: 指令列表
            
        Returns:
            预测的吞吐量
        """
        # 创建单样本数据
        sample = {
            "instructions": instructions,
            "throughput": 0.0  # 占位值
        }
        
        # 预测
        result = self.predict_from_json([sample])
        
        return result["y_pred"][0]
    
    def compute_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        计算预测指标
        
        Args:
            y_true: 真实值列表
            y_pred: 预测值列表
            
        Returns:
            指标字典
        """
        from utils.metrics import compute_regression_metrics
        
        return compute_regression_metrics(
            np.array(y_true),
            np.array(y_pred)
        )
    
    def visualize_predictions(self, y_true: List[float], y_pred: List[float], 
                             output_path: Optional[str] = None) -> None:
        """
        可视化预测结果
        
        Args:
            y_true: 真实值列表
            y_pred: 预测值列表
            output_path: 输出目录路径
        """
        from utils.visualize import plot_prediction_scatter, plot_error_histogram
        
        # 设置输出目录
        if output_path:
            os.makedirs(output_path, exist_ok=True)
        
        # 绘制预测散点图
        fig1 = plot_prediction_scatter(
            np.array(y_true),
            np.array(y_pred),
            save_path=os.path.join(output_path, "predictions.png") if output_path else None
        )
        
        # 绘制误差直方图
        errors = np.array(y_true) - np.array(y_pred)
        fig2 = plot_error_histogram(
            errors,
            save_path=os.path.join(output_path, "error_distribution.png") if output_path else None
        )
        
        if not output_path:
            import matplotlib.pyplot as plt
            plt.show()
