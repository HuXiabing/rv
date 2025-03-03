# #!/usr/bin/env python
# """
# 模型训练脚本 - 训练RISC-V吞吐量预测模型
# """
#
# import os
# import sys
# import argparse
# from pathlib import Path
#
# # 添加项目根目录到Python路径
# sys.path.append(str(Path(__file__).resolve().parent.parent))
#
# import torch
# from config import get_config
# from data import get_dataloader
# from models import get_model
# from trainers import RegressionTrainer
# from utils import set_seed, ExperimentManager
#
#
# def main():
#     parser = argparse.ArgumentParser(description="RISC-V指令吞吐量预测模型训练")
#
#     # 数据参数
#     parser.add_argument("--train_data", type=str, default="data/train_data.h5", help="训练数据路径(HDF5)")
#     parser.add_argument("--val_data", type=str, default="data/val_data.h5", help="验证数据路径(HDF5)")
#
#     # 模型参数
#     parser.add_argument("--model_type", type=str, default="transformer",
#                         choices=["transformer", "lstm", "gnn", "ithemal"], help="模型类型")
#     parser.add_argument("--embed_dim", type=int, default=128, help="嵌入维度")
#     parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
#     parser.add_argument("--num_layers", type=int, default=4, help="层数")
#     parser.add_argument("--num_heads", type=int, default=8, help="注意力头数(仅transformer)")
#     parser.add_argument("--dropout", type=float, default=0.1, help="Dropout概率")
#
#     # 训练参数
#     parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
#     parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
#     parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
#     parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
#     parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
#     parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
#     parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="优化器")
#     parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "step"], help="学习率调度器")
#
#     # 输出参数
#     parser.add_argument("--output_dir", type=str, default="experiments", help="输出目录")
#     parser.add_argument("--experiment_name", type=str, default="default", help="实验名称")
#
#     # 其他参数
#     parser.add_argument("--seed", type=int, default=42, help="随机种子")
#     parser.add_argument("--device", type=str, default=None, help="训练设备")
#     parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
#
#     args = parser.parse_args()
#
#     # 设置随机种子
#     set_seed(args.seed)
#
#     # 创建实验管理器
#     experiment_manager = ExperimentManager(args.experiment_name, args.output_dir)
#
#     # 设置设备
#     if args.device is None:
#         args.device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = torch.device(args.device)
#
#     # 创建配置
#     config = get_config(
#         model_type=args.model_type,
#         embed_dim=args.embed_dim,
#         hidden_dim=args.hidden_dim,
#         num_layers=args.num_layers,
#         num_heads=args.num_heads,
#         dropout=args.dropout,
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#         batch_size=args.batch_size,
#         epochs=args.epochs,
#         patience=args.patience,
#         clip_grad_norm=args.clip_grad_norm,
#         device=args.device,
#         train_data_path=args.train_data,
#         val_data_path=args.val_data,
#         output_dir=experiment_manager.experiment_dir,  # 使用实验目录
#         experiment_name=args.experiment_name,
#         optimizer=args.optimizer,
#         scheduler=args.scheduler,
#         seed=args.seed
#     )
#
#     # 保存配置
#     experiment_manager.save_config(config)
#
#     # 创建数据加载器
#     train_loader = get_dataloader(
#         args.train_data,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers
#     )
#
#     val_loader = get_dataloader(
#         args.val_data,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers
#     )
#
#     # 创建模型
#     model = get_model(config)
#     print(f"模型: {args.model_type.upper()}, 参数数量: {model.count_parameters():,}")
#
#     # 创建训练器
#     trainer = RegressionTrainer(model, config, experiment_manager.experiment_dir)
#
#     # 训练模型
#     print(f"Starting training..., device: {device}")
#     history = trainer.train(train_loader, val_loader)
#
#     # 保存训练历史
#     experiment_manager.history = history
#     experiment_manager.save_history()
#
#     # 打印结果
#     print(f"训练完成! 最佳验证损失: {history['best_metric']:.6f} at Epoch {history['best_epoch']+1}")
#
#     # 保存实验总结
#     experiment_manager.save_summary({
#         'model_type': args.model_type,
#         'best_val_loss': history['best_metric'],
#         'best_epoch': history['best_epoch'] + 1,
#         'parameters': model.count_parameters(),
#         'train_samples': len(train_loader.dataset),
#         'val_samples': len(val_loader.dataset)
#     })
#
#     # 完成实验
#     experiment_manager.finish()
#     print(f"Finished! Results saved to: {experiment_manager.experiment_dir}")
#
#
# if __name__ == "__main__":
#     main()

# !/usr/bin/env python
"""
模型训练脚本 - 训练RISC-V吞吐量预测模型
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from config import get_config
from data import get_dataloader
from models import get_model
from trainers import RegressionTrainer
from utils import set_seed, ExperimentManager


def main():
    parser = argparse.ArgumentParser(description="RISC-V指令吞吐量预测模型训练")

    # 数据参数
    parser.add_argument("--train_data", type=str, default="data/train_data.h5", help="训练数据路径(HDF5)")
    parser.add_argument("--val_data", type=str, default="data/val_data.h5", help="验证数据路径(HDF5)")

    # 模型参数
    parser.add_argument("--model_type", type=str, default="transformer",
                        choices=["transformer", "lstm", "gnn", "ithemal"], help="模型类型")
    parser.add_argument("--embed_dim", type=int, default=128, help="嵌入维度")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4, help="层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数(仅transformer)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout概率")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="优化器")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "step"],
                        help="学习率调度器")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="experiments", help="输出目录")
    parser.add_argument("--experiment_name", type=str, default="default", help="实验名称")
    parser.add_argument("--use_experiment_manager", action="store_true", help="是否使用实验管理器")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="训练设备")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建实验管理器
    experiment_manager = ExperimentManager(args.experiment_name, args.output_dir)

    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # 创建配置
    config = get_config(
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        clip_grad_norm=args.clip_grad_norm,
        device=args.device,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=experiment_manager.experiment_dir,  # 使用实验目录
        experiment_name=args.experiment_name,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        seed=args.seed,
        use_experiment_manager=args.use_experiment_manager
    )

    # 保存配置
    experiment_manager.save_config(config)

    # 创建数据加载器
    train_loader = get_dataloader(
        args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = get_dataloader(
        args.val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 创建模型
    model = get_model(config)
    print(f"模型: {args.model_type.upper()}, 参数数量: {model.count_parameters():,}")

    # 创建训练器
    trainer = RegressionTrainer(model, config, experiment_manager.experiment_dir)

    # 如果开启了使用实验管理器，则设置
    # if args.use_experiment_manager:
    trainer.setup_experiment(args.experiment_name, args.output_dir)

    # 训练模型
    print(f"Starting training..., device: {device}")
    history = trainer.train(train_loader, val_loader)

    # 保存训练历史
    experiment_manager.history = history
    experiment_manager.save_history()

    # 打印结果
    print(f"训练完成! 最佳验证损失: {history['best_metric']:.6f} at Epoch {history['best_epoch'] + 1}")

    # 保存实验总结
    experiment_manager.save_summary({
        'model_type': args.model_type,
        'best_val_loss': history['best_metric'],
        'best_epoch': history['best_epoch'] + 1,
        'parameters': model.count_parameters(),
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset)
    })

    # 完成实验
    experiment_manager.finish()
    print(f"Finished! Results saved to: {experiment_manager.experiment_dir}")


if __name__ == "__main__":
    main()