#!/usr/bin/env python
"""
RISC-V吞吐量预测框架 - 主入口
提供一个统一的命令行接口，用于预处理、训练、评估和推理
"""

import os
import sys
import argparse
from pathlib import Path

# 确保可以导入项目模块
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from utils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="RISC-V吞吐量预测框架",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest="command", help="请选择要执行的操作")
    
    # 预处理命令
    preprocess_parser = subparsers.add_parser("preprocess", help="预处理RISC-V指令数据")
    preprocess_parser.add_argument("--raw_data", type=str, required=True, help="原始JSON数据文件路径")
    preprocess_parser.add_argument("--output_dir", type=str, default="data", help="输出目录")
    preprocess_parser.add_argument("--max_instr_length", type=int, default=20, help="指令最大长度")
    preprocess_parser.add_argument("--max_instr_count", type=int, default=20, help="样本最大指令数量")
    preprocess_parser.add_argument("--vocab_size", type=int, default=2000, help="词汇表大小")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练RISC-V吞吐量预测模型")
    train_parser.add_argument("--model_type", type=str, default="transformer", 
                             choices=["transformer", "lstm", "gnn", "ithemal"], help="模型类型")
    train_parser.add_argument("--train_data", type=str, default="data/train_data.h5", help="训练数据路径")
    train_parser.add_argument("--val_data", type=str, default="data/val_data.h5", help="验证数据路径")
    train_parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    train_parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    train_parser.add_argument("--output_dir", type=str, default="experiments", help="输出目录")
    train_parser.add_argument("--experiment_name", type=str, default=None, help="实验名称")
    
    # 评估命令
    eval_parser = subparsers.add_parser("evaluate", help="评估RISC-V吞吐量预测模型")
    eval_parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    eval_parser.add_argument("--test_data", type=str, default="data/test_data.h5", help="测试数据路径")
    eval_parser.add_argument("--output_dir", type=str, default="evaluation", help="输出目录")
    
    # 推理命令
    predict_parser = subparsers.add_parser("predict", help="使用RISC-V吞吐量预测模型进行推理")
    predict_parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    group = predict_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_json", type=str, help="输入JSON文件路径")
    group.add_argument("--input_hdf5", type=str, help="输入HDF5文件路径")
    predict_parser.add_argument("--output", type=str, default="predictions.json", help="输出文件路径")
    
    # 恢复训练命令
    resume_parser = subparsers.add_parser("resume", help="从检查点恢复训练")
    resume_parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    resume_parser.add_argument("--additional_epochs", type=int, default=10, help="额外训练的轮数")
    resume_parser.add_argument("--experiment_name", type=str, default=None, help="新实验名称")
    
    # 全局参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="运行设备，默认自动选择")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取当前时间作为默认实验名称
    if args.command in ["train"] and args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model_type}_{timestamp}"
    
    # 执行对应的命令
    if args.command == "preprocess":
        from scripts.preprocess import main as preprocess_main
        sys.argv = [sys.argv[0]] + [
            "--raw_data", args.raw_data,
            "--output_dir", args.output_dir,
            "--max_instr_length", str(args.max_instr_length),
            "--max_instr_count", str(args.max_instr_count),
            "--vocab_size", str(args.vocab_size),
            "--seed", str(args.seed)
        ]
        preprocess_main()
    
    elif args.command == "train":
        from scripts.train import main as train_main
        sys.argv = [sys.argv[0]] + [
            "--model_type", args.model_type,
            "--train_data", args.train_data,
            "--val_data", args.val_data,
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--output_dir", args.output_dir,
            "--experiment_name", args.experiment_name,
            "--seed", str(args.seed)
        ]
        if args.device:
            sys.argv.extend(["--device", args.device])
        train_main()
    
    elif args.command == "evaluate":
        from scripts.evaluate import main as evaluate_main
        sys.argv = [sys.argv[0]] + [
            "--model_path", args.model_path,
            "--test_data", args.test_data,
            "--output_dir", args.output_dir,
            "--seed", str(args.seed)
        ]
        if args.device:
            sys.argv.extend(["--device", args.device])
        evaluate_main()
    
    elif args.command == "predict":
        from scripts.predict import main as predict_main
        sys.argv = [sys.argv[0]] + [
            "--model_path", args.model_path,
            "--output", args.output,
            "--seed", str(args.seed)
        ]
        
        if args.input_json:
            sys.argv.extend(["--input_json", args.input_json])
        elif args.input_hdf5:
            sys.argv.extend(["--input_hdf5", args.input_hdf5])
            
        if args.device:
            sys.argv.extend(["--device", args.device])
        predict_main()
    
    elif args.command == "resume":
        from scripts.resume_training import main as resume_main
        sys.argv = [sys.argv[0]] + [
            "--checkpoint", args.checkpoint,
            "--additional_epochs", str(args.additional_epochs),
            "--seed", str(args.seed)
        ]
        
        if args.experiment_name:
            sys.argv.extend(["--experiment_name", args.experiment_name])
            
        if args.device:
            sys.argv.extend(["--device", args.device])
        resume_main()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    
