#!/usr/bin/env python
"""
RISC-V throughput prediction framework - main entry
to provide a unified command line interface for preprocessing, training, evaluation and inference
"""
import os
import sys
import argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="RISC-V throughput prediction framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Select a command to execute")

    # preprocessing command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocessing RISC-V instruction")
    preprocess_parser.add_argument("--raw_data", type=str, required=True, help="Original JSON path")
    preprocess_parser.add_argument("--output_dir", type=str, default="data", help="output directory")
    preprocess_parser.add_argument("--max_instr_length", type=int, default=8, help="maximum instruction length")
    preprocess_parser.add_argument("--max_instr_count", type=int, default=400,
                                   help="maximum instruction count for a single sample")
    preprocess_parser.add_argument("--train_ratio", type=float, default=0.8, help="training set ratio")
    preprocess_parser.add_argument("--val_ratio", type=float, default=0.2, help="validation set ratio")
    preprocess_parser.add_argument("--test_ratio", type=float, default=0, help="test set ratio")

    # Incremental preprocessing command
    incremental_preprocess_parser = subparsers.add_parser("incremental_preprocess", help="Preprocessing RISC-V instruction for incremental learning")
    incremental_preprocess_parser.add_argument("--raw_data", type=str, required=True, help="new raw JSON path")
    incremental_preprocess_parser.add_argument("--existing_train_json", type=str, default="data/train_data.json",
                                               help="processed JSON data file path")
    incremental_preprocess_parser.add_argument("--existing_train_h5", type=str, default="data/train_data.h5",
                                               help="processed HDF5 data file path")
    incremental_preprocess_parser.add_argument("--output_dir", type=str, default="data", help="output directory")
    incremental_preprocess_parser.add_argument("--max_instr_length", type=int, default=8, help="maximum instruction length")
    incremental_preprocess_parser.add_argument("--max_instr_count", type=int, default=400, help="maximum instruction count for a single sample")

    # Training command
    train_parser = subparsers.add_parser("train", help="Training RISC-V throughput prediction model")
    train_parser.add_argument("--model_type", type=str, default="transformer",
                              choices=["transformer", "gnn", "lstm"], help="Model type")
    train_parser.add_argument("--train_data", type=str, default="data/train_data.h5", help="Path to training data")
    train_parser.add_argument("--val_data", type=str, default="data/val_data.h5", help="Path to validation data")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory")
    train_parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    
    # # 评估命令
    # eval_parser = subparsers.add_parser("evaluate", help="评估RISC-V吞吐量预测模型")
    # eval_parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    # eval_parser.add_argument("--test_data", type=str, default="data/test_data.h5", help="测试数据路径")
    # eval_parser.add_argument("--output_dir", type=str, default="evaluation", help="输出目录")
    #
    # # 推理命令
    # predict_parser = subparsers.add_parser("predict", help="使用RISC-V吞吐量预测模型进行推理")
    # predict_parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    # group = predict_parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--input_json", type=str, help="输入JSON文件路径")
    # group.add_argument("--input_hdf5", type=str, help="输入HDF5文件路径")
    # predict_parser.add_argument("--output", type=str, default="predictions.json", help="输出文件路径")

    # resume training command
    resume_parser = subparsers.add_parser("resume", help="resume training from a checkpoint")
    resume_parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    resume_parser.add_argument("--additional_epochs", type=int, default=10, help="extra epochs")
    resume_parser.add_argument("--experiment_name", type=str, default=None, help="new experiment name")

    # Incremental Learning Command
    incremental_parser = subparsers.add_parser("incremental",
                                               help="Perform incremental learning based on an existing model")
    incremental_parser.add_argument("--model_path", type=str, default=None,
                                    help="Path to the checkpoint of the trained model. Defaults to the most recently trained model.")
    incremental_parser.add_argument("--original", type=str, default=None,
                                    help="Original experiment directory, used to automatically find the latest model (if model_path is not specified).")
    incremental_parser.add_argument("--new_train_data", type=str, default="data/incremental_train.h5",
                                    help="Path to the newly generated training data (HDF5).")
    incremental_parser.add_argument("--val_data", type=str, default="data/val_data.h5",
                                    help="Path to the validation data (HDF5).")
    incremental_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for incremental training.")
    incremental_parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    incremental_parser.add_argument("--lr", type=float, default=5e-5,
                                    help="Learning rate, typically smaller than the initial learning rate.")
    incremental_parser.add_argument("--output_dir", type=str, default="experiments",
                                    help="Output directory.")
    incremental_parser.add_argument("--experiment_name", type=str, default=None,
                                    help="Experiment name, defaults to incremental_{original_model_name}.")
    incremental_parser.add_argument("--restart_optimizer", action="store_true",
                                    help="Whether to reinitialize the optimizer (defaults to using the optimizer state from the original model).")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on, defaults to automatic selection.")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Get the current time as the default experiment name
    if args.command in ["train"] and args.experiment_name is None:
        args.experiment_name = f"{args.model_type}"
    if args.command in ["incremental"] and args.experiment_name is None:
        args.experiment_name = "incremental"

    if args.command == "preprocess":
        from scripts.preprocess import main as preprocess_main
        sys.argv = [sys.argv[0]] + [
            "--raw_data", args.raw_data,
            "--output_dir", args.output_dir,
            "--max_instr_length", str(args.max_instr_length),
            "--max_instr_count", str(args.max_instr_count),
            "--train_ratio", str(args.train_ratio),
            "--val_ratio", str(args.val_ratio),
            "--test_ratio", str(args.test_ratio),
            "--seed", str(args.seed)
        ]
        preprocess_main()

    elif args.command == "incremental_preprocess":
        from scripts.incremental_preprocess import main as incremental_preprocess_main
        sys.argv = [sys.argv[0]] + [
            "--raw_data", args.raw_data,
            "--existing_train_json", args.existing_train_json,
            "--existing_train_h5", args.existing_train_h5,
            "--output_dir", args.output_dir,
            "--max_instr_length", str(args.max_instr_length),
            "--max_instr_count", str(args.max_instr_count),
            "--seed", str(args.seed)
        ]
        incremental_preprocess_main()

    
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
    
    # elif args.command == "evaluate":
    #     from scripts.evaluate import main as evaluate_main
    #     sys.argv = [sys.argv[0]] + [
    #         "--model_path", args.model_path,
    #         "--test_data", args.test_data,
    #         "--output_dir", args.output_dir,
    #         "--seed", str(args.seed)
    #     ]
    #     if args.device:
    #         sys.argv.extend(["--device", args.device])
    #     evaluate_main()
    #
    # elif args.command == "predict":
    #     from scripts.predict import main as predict_main
    #     sys.argv = [sys.argv[0]] + [
    #         "--model_path", args.model_path,
    #         "--output", args.output,
    #         "--seed", str(args.seed)
    #     ]
    #
    #     if args.input_json:
    #         sys.argv.extend(["--input_json", args.input_json])
    #     elif args.input_hdf5:
    #         sys.argv.extend(["--input_hdf5", args.input_hdf5])
    #
    #     if args.device:
    #         sys.argv.extend(["--device", args.device])
    #     predict_main()
    #
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

    elif args.command == "incremental":
        from scripts.incremental_learning import main as incremental_main
        sys.argv = [sys.argv[0]] + [
            "--new_train_data", args.new_train_data,
            "--val_data", args.val_data,
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--output_dir", args.output_dir,
            "--seed", str(args.seed)
        ]

        if args.model_path:
            sys.argv.extend(["--model_path", args.model_path])
        if args.original:
            sys.argv.extend(["--original", args.original])
        if args.experiment_name:
            sys.argv.extend(["--experiment_name", args.experiment_name])
        if args.restart_optimizer:
            sys.argv.append("--restart_optimizer")
        if args.device:
            sys.argv.extend(["--device", args.device])

        incremental_main()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    
