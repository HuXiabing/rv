import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional
import concurrent.futures
from tqdm import tqdm

from .predictor import RISCVPredictor


class BatchPredictor:
    """批量处理大量样本的预测器"""
    
    def __init__(self, model_path: str, device: Optional[str] = None, batch_size: int = 32):
        """
        初始化批量预测器
        
        Args:
            model_path: 模型检查点路径
            device: 推理设备
            batch_size: 批处理大小
        """
        self.predictor = RISCVPredictor(model_path, device)
        self.batch_size = batch_size
    
    def predict_file(self, input_file: str, output_file: str,
                    compute_metrics: bool = True) -> Dict[str, Any]:
        """
        处理整个文件的预测
        
        Args:
            input_file: 输入文件路径 (JSON或HDF5)
            output_file: 输出JSON文件路径
            compute_metrics: 是否计算指标
            
        Returns:
            预测结果和指标
        """
        print(f"正在处理文件: {input_file}")
        
        # 根据文件类型选择不同的预测方法
        if input_file.endswith('.h5'):
            result = self.predictor.predict_from_hdf5(input_file, self.batch_size)
        else:
            result = self.predictor.predict_from_json(input_file, self.batch_size)
        
        # 计算评估指标
        if compute_metrics and "y_true" in result:
            metrics = self.predictor.compute_metrics(result["y_true"], result["y_pred"])
            result["metrics"] = metrics
            
            # 打印主要指标
            print("\n===== 预测指标 =====")
            for name, value in metrics.items():
                print(f"{name}: {value:.6f}")
        
        # 保存结果
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"预测结果已保存到: {output_file}")
        
        return result
    
    def predict_directory(self, input_dir: str, output_dir: str, 
                         file_pattern: str = "*.json",
                         parallel: bool = False,
                         max_workers: int = 4) -> Dict[str, Dict[str, Any]]:
        """
        处理目录中的所有文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            file_pattern: 文件匹配模式
            parallel: 是否并行处理
            max_workers: 最大工作进程数
            
        Returns:
            所有文件的预测结果
        """
        from pathlib import Path
        
        # 获取所有匹配的文件
        input_files = list(Path(input_dir).glob(file_pattern))
        print(f"找到 {len(input_files)} 个文件匹配 '{file_pattern}'")
        
        # 准备输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备结果
        all_results = {}
        
        if parallel and len(input_files) > 1:
            print(f"使用 {max_workers} 个工作进程并行处理文件")
            
            # 定义工作函数
            def process_file(input_file):
                try:
                    output_file = os.path.join(output_dir, input_file.name.replace('.json', '_predictions.json'))
                    result = self.predict_file(str(input_file), output_file)
                    return input_file.name, result
                except Exception as e:
                    print(f"处理文件 {input_file} 时出错: {e}")
                    return input_file.name, {"error": str(e)}
            
            # 并行处理文件
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_file, f): f for f in input_files}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理文件"):
                    file_name, result = future.result()
                    all_results[file_name] = result
        else:
            # 顺序处理文件
            for input_file in tqdm(input_files, desc="处理文件"):
                output_file = os.path.join(output_dir, input_file.name.replace('.json', '_predictions.json'))
                result = self.predict_file(str(input_file), output_file)
                all_results[input_file.name] = result
        
        # 保存汇总结果
        with open(os.path.join(output_dir, "all_predictions.json"), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
