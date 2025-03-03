#!/usr/bin/env python
"""
模型评估脚本 - 评估RISC-V吞吐量预测模型
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from config import Config
from data import get_dataloader
from models import BaseModel
from utils import (
    compute_regression_metrics, 
    compute_error_distribution,
    create_training_report,
    set_seed
)


def main():
    parser = argparse.ArgumentParser(description="RISC-V指令吞吐量预测模型评估")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    
    # 数据参数
    parser.add_argument("--test_data", type=str, default="data/test_data.h5", help="测试数据路径(HDF5)")
    
    # 评估参数
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="evaluation", help="评估结果输出目录")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="评估设备")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = BaseModel.load(args.model_path, device=device)
    model.eval()
    
    # 创建数据加载器
    test_loader = get_dataloader(
        args.test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 评估模型
    print(f"在测试集上评估模型，设备: {device}")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 获取数据
            x = batch['X'].to(device)
            instruction_count = batch['instruction_count'].to(device)
            y = batch['Y'].to(device)
            
            # 前向传播
            output = model(x, instruction_count)
            
            # 保存预测和真实值
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 计算评估指标
    metrics = compute_regression_metrics(all_targets, all_preds)
    error_distribution = compute_error_distribution(all_targets, all_preds)
    
    # 打印指标
    print("\n===== 评估指标 =====")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    
    # 保存预测结果
    predictions = {
        'y_true': all_targets.tolist(),
        'y_pred': all_preds.tolist()
    }
    
    with open(os.path.join(args.output_dir, "predictions.json"), 'w') as f:
        json.dump(predictions, f)
    
    # 保存评估指标
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 保存误差分布
    error_stats = {
        'error_stats': error_distribution['error_stats'],
        'rel_error_stats': error_distribution['rel_error_stats']
    }
    
    with open(os.path.join(args.output_dir, "error_stats.json"), 'w') as f:
        json.dump(error_stats, f, indent=2)
    
    # 创建评估报告
    create_training_report(
        {'train_losses': [], 'val_losses': []},  # 没有训练历史，使用空列表
        all_targets,
        all_preds,
        metrics,
        args.output_dir,
        "evaluation_report"
    )
    
    print(f"评估完成! 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
