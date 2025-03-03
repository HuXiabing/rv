#!/usr/bin/env python
"""
恢复训练脚本 - 从检查点恢复RISC-V吞吐量预测模型训练
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from config import Config
from data import get_dataloader
from models import get_model
from trainers import RegressionTrainer
from utils import set_seed, ExperimentManager


def main():
    parser = argparse.ArgumentParser(description="从检查点恢复RISC-V吞吐量预测模型训练")
    
    # 检查点参数
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    
    # 数据参数
    parser.add_argument("--train_data", type=str, default="data/train_data.h5", help="训练数据路径(HDF5)")
    parser.add_argument("--val_data", type=str, default="data/val_data.h5", help="验证数据路径(HDF5)")
    
    # 训练参数
    parser.add_argument("--additional_epochs", type=int, default=10, help="额外训练的轮数")
    
    # 输出参数
    parser.add_argument("--experiment_name", type=str, help="新实验名称，不指定则使用原实验名称并添加'_continued'后缀")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="训练设备")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载检查点
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取配置
    config_dict = checkpoint.get('config', None)
    if config_dict is None:
        raise ValueError(f"检查点 {args.checkpoint} 不包含配置信息")
    
    config = Config(**config_dict)
    
    # 更新设备
    if args.device:
        config.device = args.device
    device = torch.device(config.device)
    
    # 创建新实验名称
    if args.experiment_name:
        new_experiment_name = args.experiment_name
    else:
        old_experiment_name = config.experiment_name
        new_experiment_name = f"{old_experiment_name}_continued"
    
    # 获取原始总轮数
    total_epochs = checkpoint.get('epoch', 0) + 1 + args.additional_epochs
    
    # 创建实验管理器
    experiment_manager = ExperimentManager(new_experiment_name, config.output_dir)
    
    # 更新配置
    config.experiment_name = new_experiment_name
    config.experiment_dir = experiment_manager.experiment_dir
    config.checkpoint_dir = experiment_manager.checkpoint_dir
    config.log_dir = experiment_manager.log_dir
    config.epochs = total_epochs
    
    # 保存配置
    experiment_manager.save_config(config)
    
    # 创建数据加载器
    train_loader = get_dataloader(
        args.train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        args.val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 创建模型
    model = get_model(config)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    
    # 创建训练器
    trainer = RegressionTrainer(model, config, experiment_manager.experiment_dir)
    
    # 初始化训练器组件
    if hasattr(trainer, 'optimizer') and 'optimizer_state' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    if hasattr(trainer, 'scheduler') and 'scheduler_state' in checkpoint:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # 恢复训练历史
    trainer.start_epoch = checkpoint.get('epoch', 0) + 1
    trainer.global_step = checkpoint.get('global_step', 0)
    trainer.best_metric = checkpoint.get('best_metric', float('inf'))
    trainer.best_epoch = checkpoint.get('best_epoch', 0)
    trainer.train_losses = checkpoint.get('train_losses', [])
    trainer.val_losses = checkpoint.get('val_losses', [])
    trainer.learning_rates = checkpoint.get('learning_rates', [])
    
    # 打印恢复信息
    print(f"从检查点 {args.checkpoint} 恢复训练")
    print(f"原始轮数: {trainer.start_epoch}，额外训练轮数: {args.additional_epochs}")
    print(f"当前最佳验证损失: {trainer.best_metric:.6f} at Epoch {trainer.best_epoch+1}")
    
    # 训练模型
    print(f"继续训练，设备: {device}")
    history = trainer.train(train_loader, val_loader)
    
    # 保存训练历史
    experiment_manager.history = history
    experiment_manager.save_history()
    
    # 打印结果
    print(f"训练完成! 最佳验证损失: {history['best_metric']:.6f} at Epoch {history['best_epoch']+1}")
    
    # 保存实验总结
    experiment_manager.save_summary({
        'model_type': config.model_type,
        'best_val_loss': history['best_metric'],
        'best_epoch': history['best_epoch'] + 1,
        'parameters': model.count_parameters(),
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'original_checkpoint': str(checkpoint_path),
        'additional_epochs': args.additional_epochs
    })
    
    # 完成实验
    experiment_manager.finish()
    print(f"恢复训练完成! 结果保存在: {experiment_manager.experiment_dir}")


if __name__ == "__main__":
    main()
