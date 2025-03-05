# #!/usr/bin/env python
# """
# 增量学习脚本 - 基于已有模型进行增量学习
# """
#
# import os
# import sys
# import argparse
# import glob
# from pathlib import Path
#
# # 添加项目根目录到Python路径
# sys.path.append(str(Path(__file__).resolve().parent.parent))
#
# import torch
# from config import Config
# from models import get_model
# from data import get_dataloader
# from trainers import RegressionTrainer
# from utils import set_seed, ExperimentManager
#
#
# def find_latest_model(experiment_dir):
#     """
#     在指定实验目录中查找最新的模型检查点
#
#     Args:
#         experiment_dir: 实验目录路径
#
#     Returns:
#         最新模型的路径，如果找不到则返回None
#     """
#     checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
#
#     # 首先尝试查找最佳模型
#     best_model_path = os.path.join(checkpoint_dir, "model_best.pth")
#     if os.path.exists(best_model_path):
#         return best_model_path
#
#     # 如果找不到最佳模型，查找最新的检查点
#     latest_model_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
#     if os.path.exists(latest_model_path):
#         return latest_model_path
#
#     # 如果都找不到，查找最新的epoch检查点
#     checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
#     if checkpoint_files:
#         # 按文件修改时间排序
#         checkpoint_files.sort(key=os.path.getmtime, reverse=True)
#         return checkpoint_files[0]
#
#     return None
#
#
# def main():
#     parser = argparse.ArgumentParser(description="基于已有模型的增量学习")
#
#     # 模型参数
#     parser.add_argument("--model_path", type=str, default=None,
#                         help="已训练模型的检查点路径，默认使用最新训练的模型")
#     parser.add_argument("--experiment_dir", type=str, default="experiments",
#                         help="原实验目录，用于自动查找最新模型（如果未指定model_path）")
#
#     # 数据参数
#     parser.add_argument("--new_train_data", type=str, required=True, help="新生成的训练数据路径(HDF5)")
#     parser.add_argument("--val_data", type=str, default="data/val_data.h5", help="验证数据路径(HDF5)")
#
#     # 训练参数
#     parser.add_argument("--epochs", type=int, default=10, help="增量训练的轮数")
#     parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
#     parser.add_argument("--lr", type=float, default=5e-5, help="学习率，通常比初始学习率小")
#     parser.add_argument("--weight_decay", type=float, default=1e-6, help="权重衰减")
#
#     # 输出参数
#     parser.add_argument("--output_dir", type=str, default="experiments", help="输出目录")
#     parser.add_argument("--experiment_name", type=str, default=None,
#                         help="实验名称，默认为incremental_{原模型名称}")
#
#     # 其他参数
#     parser.add_argument("--seed", type=int, default=42, help="随机种子")
#     parser.add_argument("--device", type=str, default=None, help="运行设备")
#     parser.add_argument("--restart_optimizer", action="store_true",
#                         help="是否重新初始化优化器（默认使用原模型的优化器状态）")
#     parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
#
#     args = parser.parse_args()
#
#     # 设置随机种子
#     set_seed(args.seed)
#
#     # 设置设备
#     if args.device is None:
#         args.device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = torch.device(args.device)
#
#     # 如果没有指定模型路径但指定了实验目录，自动查找最新模型
#     if args.model_path is None:
#         if args.experiment_dir is None:
#             # 如果两者都未指定，查找experiments目录下最新的实验
#             if os.path.exists("experiments"):
#                 experiments = [d for d in os.listdir("experiments")
#                                if os.path.isdir(os.path.join("experiments", d))]
#                 if experiments:
#                     # 按修改时间排序
#                     experiments.sort(key=lambda d: os.path.getmtime(os.path.join("experiments", d)),
#                                      reverse=True)
#                     args.experiment_dir = os.path.join("experiments", experiments[0])
#                     print(f"自动选择最新实验目录: {args.experiment_dir}")
#                 else:
#                     parser.error("找不到任何实验目录，请指定 --model_path 或 --experiment_dir")
#
#         # 从实验目录查找最新模型
#         if args.experiment_dir:
#             model_path = find_latest_model(args.experiment_dir)
#             if model_path:
#                 args.model_path = model_path
#                 print(f"使用最新模型: {args.model_path}")
#             else:
#                 parser.error(f"在 {args.experiment_dir} 中找不到模型检查点")
#
#     # 加载模型检查点
#     print(f"加载模型 {args.model_path}")
#     checkpoint = torch.load(args.model_path, map_location='cpu')
#
#     # 获取配置
#     config_dict = checkpoint.get('config', None)
#     if config_dict is None:
#         raise ValueError(f"检查点 {args.model_path} 不包含配置信息")
#
#     # 创建配置对象
#     config = Config(**config_dict)
#
#     # 更新配置
#     config.device = args.device
#     config.lr = args.lr
#     config.epochs = args.epochs
#     config.batch_size = args.batch_size
#     config.weight_decay = args.weight_decay
#
#     # 创建实验名称
#     if args.experiment_name is None:
#         original_experiment = os.path.basename(os.path.dirname(args.model_path))
#         args.experiment_name = f"incremental_{original_experiment}"
#
#     # 创建实验管理器
#     experiment_manager = ExperimentManager(args.experiment_name, args.output_dir)
#
#     # 更新配置目录
#     config.experiment_name = args.experiment_name
#     config.experiment_dir = experiment_manager.experiment_dir
#     config.checkpoint_dir = experiment_manager.checkpoint_dir
#     config.log_dir = experiment_manager.log_dir
#
#     # 保存配置
#     experiment_manager.save_config(config)
#
#     # 创建模型
#     model = get_model(config)
#
#     # 加载模型权重
#     model.load_state_dict(checkpoint['model_state'])
#     model.to(device)
#
#     # 输出模型信息
#     print(f"模型: {config.model_type.upper()}, 参数数量: {model.count_parameters():,}")
#
#     # 创建数据加载器
#     train_loader = get_dataloader(
#         args.new_train_data,
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
#     # 创建训练器
#     trainer = RegressionTrainer(model, config, experiment_manager.experiment_dir, experiment_manager)
#
#     # 如果不需要重新初始化优化器，则加载之前的优化器状态
#     if not args.restart_optimizer and 'optimizer_state' in checkpoint:
#         try:
#             trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
#             print("已加载原模型的优化器状态")
#         except Exception as e:
#             print(f"加载优化器状态失败: {e}")
#             print("使用新初始化的优化器")
#     else:
#         print("使用新初始化的优化器")
#
#     # 开始增量训练
#     print(f"开始增量训练，设备: {device}")
#     print(f"训练数据: {args.new_train_data}, 样本数: {len(train_loader.dataset)}")
#     print(f"验证数据: {args.val_data}, 样本数: {len(val_loader.dataset)}")
#     print(f"训练轮数: {args.epochs}, 学习率: {args.lr}")
#
#     # 训练模型
#     history = trainer.train(train_loader, val_loader)
#
#     # 保存训练历史
#     experiment_manager.history = history
#     experiment_manager.save_history()
#
#     # 记录增量学习相关信息
#     summary = {
#         'model_type': config.model_type,
#         'original_model': args.model_path,
#         'train_samples': len(train_loader.dataset),
#         'val_samples': len(val_loader.dataset),
#         'best_val_loss': history.get('best_metric', 0),
#         'best_epoch': history.get('best_epoch', 0) + 1,
#         'incremental_learning': True
#     }
#
#     # 保存实验摘要
#     experiment_manager.save_summary(summary)
#
#     # 完成实验
#     experiment_manager.finish()
#     print(f"增量学习完成! 模型保存在: {experiment_manager.experiment_dir}")
#     print(f"最佳验证损失: {history.get('best_metric', 0):.6f} at Epoch {history.get('best_epoch', 0) + 1}")
#
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
"""
增量训练脚本 - 基于已有模型进行增量训练
"""

#!/usr/bin/env python
"""
增量训练脚本 - 基于已有模型进行增量训练
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from config import get_config
from data import get_dataloader
from models import get_model
from trainers import RegressionTrainer
from utils import set_seed, ExperimentManager


def find_latest_model(experiment_dir):
    """
    在指定实验目录中查找最新的模型检查点
    """
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    # 首先尝试查找最佳模型
    best_model_path = os.path.join(checkpoint_dir, "model_best.pth")
    if os.path.exists(best_model_path):
        return best_model_path

    # 如果找不到最佳模型，查找最新的检查点
    latest_model_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    if os.path.exists(latest_model_path):
        return latest_model_path

    # 如果都找不到，查找最新的epoch检查点
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if checkpoint_files:
        # 按文件修改时间排序
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        return checkpoint_files[0]

    return None


def main():
    parser = argparse.ArgumentParser(description="基于已有模型的增量训练")

    # 模型参数
    parser.add_argument("--model_path", type=str, default=None,
                        help="已训练模型的检查点路径，默认使用最新训练的模型")
    parser.add_argument("--experiment_dir", type=str, default="experiments",
                        help="原实验目录，用于自动查找最新模型（如果未指定model_path）")
    parser.add_argument("--model_type", type=str, default="transformer",
                        choices=["transformer", "lstm", "gnn", "ithemal"], help="模型类型")

    # 数据参数
    parser.add_argument("--new_train_data", type=str, required=True, help="新的训练数据路径(HDF5)")
    parser.add_argument("--val_data", type=str, default="data/val_data.h5", help="验证数据路径(HDF5)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=10, help="增量训练的轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="权重衰减")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--restart_optimizer", action="store_true", help="是否重新初始化优化器")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="experiments", help="输出目录")
    parser.add_argument("--experiment_name", type=str, default="incremental_training", help="实验名称")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="运行设备")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # 如果没有指定模型路径但指定了实验目录，自动查找最新模型
    if args.model_path is None:
        if args.experiment_dir is None:
            # 如果两者都未指定，查找experiments目录下最新的实验
            if os.path.exists("experiments"):
                experiments = [d for d in os.listdir("experiments")
                               if os.path.isdir(os.path.join("experiments", d))]
                if experiments:
                    # 按修改时间排序
                    experiments.sort(key=lambda d: os.path.getmtime(os.path.join("experiments", d)),
                                     reverse=True)
                    args.experiment_dir = os.path.join("experiments", experiments[0])
                    print(f"自动选择最新实验目录: {args.experiment_dir}")
                else:
                    parser.error("找不到任何实验目录，请指定 --model_path 或 --experiment_dir")

        # 从实验目录查找最新模型
        if args.experiment_dir:
            model_path = find_latest_model(args.experiment_dir)
            if model_path:
                args.model_path = model_path
                print(f"使用最新模型: {args.model_path}")
            else:
                parser.error(f"在 {args.experiment_dir} 中找不到模型检查点")

    # 加载模型检查点
    print(f"加载模型 {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')

    # 从检查点中加载配置
    config_dict = checkpoint.get('config', None)
    if config_dict is None:
        raise ValueError(f"检查点 {args.model_path} 不包含配置信息")

    # 创建配置对象
    config = get_config(**config_dict)

    # 创建实验管理器
    experiment_manager = ExperimentManager(args.experiment_name, args.output_dir)
    experiment_manager.save_config(config)

    # 创建模型
    model = get_model(config)

    # 部分加载权重
    model_state_dict = model.state_dict()
    for name, param in checkpoint['model_state'].items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"跳过参数 {name}: 形状不匹配 ({param.shape} vs {model_state_dict[name].shape})")
        else:
            print(f"跳过参数 {name}: 不在当前模型中")

    # 加载更新后的状态字典
    model.load_state_dict(model_state_dict)
    model.to(device)

    # 输出模型信息
    print(f"模型: {config.model_type.upper()}, 参数数量: {model.count_parameters():,}")

    # 创建数据加载器
    train_loader = get_dataloader(
        args.new_train_data,
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

    # 创建训练器
    trainer = RegressionTrainer(model, config, experiment_manager.experiment_dir, experiment_manager)

    # 加载优化器状态（如果不重新初始化）
    if not args.restart_optimizer and 'optimizer_state' in checkpoint:
        try:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("已加载原模型的优化器状态")
        except Exception as e:
            print(f"加载优化器状态失败: {e}")
            print("使用新初始化的优化器")
    else:
        print("使用新初始化的优化器")

    # 开始增量训练
    print(f"开始增量训练，设备: {device}")
    print(f"训练数据: {args.new_train_data}, 样本数: {len(train_loader.dataset)}")
    print(f"验证数据: {args.val_data}, 样本数: {len(val_loader.dataset)}")
    print(f"训练轮数: {args.epochs}, 学习率: {args.lr}")

    # 训练模型
    history = trainer.train(train_loader, val_loader)

    # 保存训练历史
    experiment_manager.history = history
    experiment_manager.save_history()

    # 保存实验总结
    experiment_manager.save_summary({
        'model_type': config.model_type,
        'best_val_loss': history['best_metric'],
        'best_epoch': history['best_epoch'] + 1,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'incremental_learning': True
    })

    # 完成实验
    experiment_manager.finish()
    print(f"增量训练完成! 模型保存在: {experiment_manager.experiment_dir}")
    print(f"最佳验证损失: {history['best_metric']:.6f} at Epoch {history['best_epoch'] + 1}")


if __name__ == "__main__":
    main()