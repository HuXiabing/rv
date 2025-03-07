#!/usr/bin/env python
"""
模型推理脚本 - 使用RISC-V吞吐量预测模型进行推理
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
from models import BaseModel
from data import RISCVDataProcessor
from utils import set_seed, compute_regression_metrics
from inference_ import RISCVPredictor, BatchPredictor


def main():
    parser = argparse.ArgumentParser(description="RISC-V指令吞吐量预测")

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")

    # 输入参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_json", type=str, help="输入JSON文件路径")
    group.add_argument("--input_hdf5", type=str, help="输入HDF5文件路径")
    group.add_argument("--input_dir", type=str, help="输入目录，包含多个JSON或HDF5文件")

    # 输出参数
    parser.add_argument("--output", type=str, default="predictions.json", help="预测结果输出文件或目录")
    parser.add_argument("--visualize", action="store_true", help="可视化预测结果")

    # 其他参数
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="推理设备")
    parser.add_argument("--parallel", action="store_true", help="并行处理多个文件（当使用--input_dir时）")
    parser.add_argument("--workers", type=int, default=4, help="并行处理的工作进程数")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 处理单个文件
    if args.input_json or args.input_hdf5:
        input_file = args.input_json or args.input_hdf5

        # 创建预测器
        predictor = RISCVPredictor(args.model_path, args.device)

        # 根据文件类型预测
        if input_file.endswith('.h5'):
            result = predictor.predict_from_hdf5(input_file, args.batch_size)
        else:
            result = predictor.predict_from_json(input_file, args.batch_size)

        # 计算指标（如果有真实值）
        if "y_true" in result:
            metrics = predictor.compute_metrics(result["y_true"], result["y_pred"])
            result["metrics"] = metrics

            # 打印指标
            print("\n===== 预测指标 =====")
            for name, value in metrics.items():
                print(f"{name}: {value:.6f}")

            # 可视化结果
            if args.visualize:
                output_dir = os.path.dirname(os.path.abspath(args.output))
                predictor.visualize_predictions(
                    result["y_true"],
                    result["y_pred"],
                    output_dir
                )

        # 保存结果
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"预测结果已保存到: {args.output}")

    # 处理目录
    elif args.input_dir:
        # 创建批量预测器
        batch_predictor = BatchPredictor(args.model_path, args.device, args.batch_size)

        # 预测目录中的所有文件
        file_pattern = "*.json" if not args.input_dir.endswith('.h5') else "*.h5"
        results = batch_predictor.predict_directory(
            args.input_dir,
            args.output,
            file_pattern=file_pattern,
            parallel=args.parallel,
            max_workers=args.workers
        )

        print(f"已处理 {len(results)} 个文件，结果保存在: {args.output}")


if __name__ == "__main__":
    main()