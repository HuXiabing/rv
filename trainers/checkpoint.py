import os
import torch
import glob
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path


class CheckpointManager:
    """管理模型检查点的保存和加载"""

    def __init__(self,
                 checkpoint_dir: str,
                 model,
                 optimizer=None,
                 scheduler=None,
                 max_checkpoints: int = 5):
        """
        初始化检查点管理器

        Args:
            checkpoint_dir: 检查点保存目录
            model: 要保存的模型
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
            max_checkpoints: 保留的最大检查点数量
        """
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_checkpoints = max_checkpoints

        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self,
             epoch: int,
             metrics: Dict[str, float],
             is_best: bool = False,
             additional_data: Optional[Dict[str, Any]] = None) -> str:
        """
        保存模型检查点

        Args:
            epoch: 当前训练轮次
            metrics: 包含训练和验证指标的字典
            is_best: 是否为最佳模型
            additional_data: 要保存的额外数据

        Returns:
            检查点路径
        """
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'metrics': metrics
        }

        # 添加优化器状态
        if self.optimizer:
            checkpoint['optimizer_state'] = self.optimizer.state_dict()

        # 添加调度器状态
        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()

        # 添加额外数据
        if additional_data:
            checkpoint.update(additional_data)

        # 保存常规检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, checkpoint_path)

        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)

        # 如果是最佳模型，保存一个额外的副本
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
            torch.save(checkpoint, best_path)

        # 删除旧检查点
        self.prune_old_checkpoints()

        return checkpoint_path

    def load(self,
             checkpoint_path: Optional[str] = None,
             best: bool = False,
             device=None) -> Dict[str, Any]:
        """
        加载模型检查点

        Args:
            checkpoint_path: 特定检查点路径，若不指定则加载最新
            best: 是否加载最佳模型
            device: 加载到的设备

        Returns:
            加载的检查点数据
        """
        # 确定要加载的检查点路径
        if checkpoint_path is None:
            if best:
                checkpoint_path = os.path.join(self.checkpoint_dir, "model_best.pth")
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"找不到最佳模型检查点: {checkpoint_path}")
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
                if not os.path.exists(checkpoint_path):
                    checkpoint_paths = self.get_all_checkpoints()
                    if not checkpoint_paths:
                        raise FileNotFoundError(f"在 {self.checkpoint_dir} 中找不到任何检查点")
                    checkpoint_path = checkpoint_paths[-1]  # 最新的检查点

        # 确定设备
        if device is None:
            device = next(self.model.parameters()).device

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state'])

        # 加载优化器状态
        if self.optimizer and 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        # 加载调度器状态
        if self.scheduler and 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        print(f"从 {checkpoint_path} 加载检查点")
        return checkpoint

    def prune_old_checkpoints(self):
        """删除旧的检查点，仅保留最近的几个"""
        # 获取所有常规检查点路径
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")
        checkpoint_paths = glob.glob(checkpoint_pattern)

        # 过滤掉特殊检查点
        checkpoint_paths = [p for p in checkpoint_paths
                            if "latest" not in p and "best" not in p]

        # 如果检查点数量不超过限制，不做任何操作
        if len(checkpoint_paths) <= self.max_checkpoints:
            return

        # 按修改时间排序
        checkpoint_paths.sort(key=lambda x: os.path.getmtime(x))

        # 删除最旧的检查点，直到数量符合限制
        for path in checkpoint_paths[:-self.max_checkpoints]:
            os.remove(path)
            print(f"删除旧检查点: {path}")

    def get_all_checkpoints(self) -> List[str]:
        """获取所有检查点路径"""
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")
        checkpoint_paths = glob.glob(checkpoint_pattern)

        # 过滤掉特殊检查点
        checkpoint_paths = [p for p in checkpoint_paths
                            if "latest" not in p and "best" not in p]

        # 按修改时间排序
        checkpoint_paths.sort(key=lambda x: os.path.getmtime(x))

        return checkpoint_paths

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新检查点的路径"""
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
        if os.path.exists(latest_path):
            return latest_path

        checkpoint_paths = self.get_all_checkpoints()
        if checkpoint_paths:
            return checkpoint_paths[-1]  # 最新的检查点

        return None

    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点的路径"""
        best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
        if os.path.exists(best_path):
            return best_path

        return None