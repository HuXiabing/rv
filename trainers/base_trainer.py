import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from utils.experiment import ExperimentManager
from tqdm import tqdm
import matplotlib.pyplot as plt


class BaseTrainer:

    def __init__(self, model, config, experiment_dir=None):

        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.start_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0

        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        self.experiment_dir = experiment_dir or config.experiment_dir
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # initialize optimizer, scheduler, criterion and metric function
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metric_fn = None
        self.clip_grad_norm = config.clip_grad_norm
        self.experiment = None

    def setup_experiment(self, experiment_name=None, base_dir=None):

        experiment_name = experiment_name or getattr(self.config, 'experiment_name', 'unnamed_experiment')
        base_dir = base_dir or getattr(self.config, 'experiment_base_dir', 'experiments')

        self.experiment = ExperimentManager(experiment_name, base_dir)
        self.experiment.save_config(self.config)

        return self.experiment

    def setup_optimizer(self):
        raise NotImplementedError("Implemented in child class")

    def setup_scheduler(self):
        raise NotImplementedError("Implemented in child class")

    def setup_criterion(self):
        raise NotImplementedError("Implemented in child class")

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        训练一个周期

        Args:
            train_loader: 训练数据加载器

        Returns:
            包含训练指标的字典
        """
        self.model.train()
        total_loss = 0.0

        # 进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            x = batch['X'].to(self.device)
            instruction_count = batch['instruction_count'].to(self.device)
            y = batch['Y'].to(self.device)

            # 清除梯度
            self.optimizer.zero_grad()

            # 前向传播
            # output = self.model(x, instruction_count)
            output = self.model(x)

            # 计算损失
            loss = self.criterion(output, y)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            # 更新参数
            self.optimizer.step()

            # 更新总损失
            total_loss += loss.item()

            # 更新进度条
            progress_bar.set_postfix({"loss": loss.item()})

            # 更新全局步数
            self.global_step += 1

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)

        # 记录当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

        return {
            "loss": avg_loss,
            "lr": current_lr
        }

    def validate(self, val_loader) -> Dict[str, float]:
        """
        在验证集上评估模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            包含验证指标的字典
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # 获取数据
                x = batch['X'].to(self.device)
                instruction_count = batch['instruction_count'].to(self.device)
                y = batch['Y'].to(self.device)

                # 前向传播
                output = self.model(x, instruction_count)

                # 计算损失
                loss = self.criterion(output, y)

                # 更新总损失
                total_loss += loss.item()

                # 收集预测值和真实值
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        # 计算平均损失
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)

        # 计算其他指标
        metrics = {
            "loss": avg_loss
        }

        if self.metric_fn:
            additional_metrics = self.metric_fn(
                np.array(all_targets),
                np.array(all_preds)
            )
            metrics.update(additional_metrics)

        return metrics

    def train(self, train_loader, val_loader, num_epochs=None, resume=False, checkpoint_path=None):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数，如果为None则使用config中的epochs
            resume: 是否从检查点恢复训练
            checkpoint_path: 检查点路径，如果为None且resume为True则加载最新检查点

        Returns:
            训练历史
        """
        # 设置训练轮数
        num_epochs = num_epochs or self.config.epochs

        # 如果需要恢复训练
        if resume:
            self._resume_checkpoint(checkpoint_path)

        # 记录开始时间
        start_time = time.time()

        print(f"开始训练: 从Epoch {self.start_epoch + 1} 到 {num_epochs}")

        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch

            # 训练一个周期
            train_metrics, train_batch_result = self.train_epoch(train_loader)

            # 在验证集上评估
            val_metrics = self.validate(val_loader)

            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # 检查是否是最佳模型
            is_best = val_metrics["loss"] < self.best_metric
            if is_best:
                self.best_metric = val_metrics["loss"]
                self.best_epoch = epoch
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # 保存检查点
            if (epoch + 1) % self.config.save_freq == 0 or is_best:
                self._save_checkpoint(epoch, train_metrics, val_metrics, is_best)

            # 保存指令类型和基本块长度的统计数据
            instruction_stats = {
                    "instruction_avg_loss": train_batch_result.get_instruction_avg_loss(),
                    "instruction_counts": train_batch_result.instruction_counts
            }

            block_length_stats = {
                "block_length_avg_loss": train_batch_result.get_block_length_avg_loss(),
                "block_length_counts": train_batch_result.block_lengths_counts
            }
            # 使用实验管理器保存这些统计数据（需要添加到experiment.py）
            if hasattr(self, 'experiment') and self.experiment:
                self.experiment.save_instruction_stats(instruction_stats, epoch)
                self.experiment.save_block_length_stats(block_length_stats, epoch)

                # 生成可视化
                self.experiment.visualize_epoch_stats(
                    instruction_stats,
                    block_length_stats,
                    epoch
                )

            # 打印进度
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_metrics['loss']:.6f} - "
                  f"Val Loss: {val_metrics['loss']:.6f}" +
                  (f" - Best Val Loss: {self.best_metric:.6f} (Epoch {self.best_epoch + 1})" if is_best else ""))

            # 绘制训练进度
            # if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            #     self._plot_progress()

            # 早停
            if self.early_stopping_counter >= self.config.patience:
                print(f"早停: 验证损失在{self.config.patience}个周期内没有改善")
                break

        # 训练完成
        training_time = time.time() - start_time
        print(f"训练完成! 总时间: {training_time:.2f} 秒")
        print(f"最佳验证损失: {self.best_metric:.6f} at Epoch {self.best_epoch + 1}")

        # # 绘制最终训练进度
        # self._plot_progress()

        # 加载最佳模型
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "model_best.pth")
        if os.path.exists(best_checkpoint_path):
            self._resume_checkpoint(best_checkpoint_path, only_model=True)
            print(f"已加载最佳模型 (Epoch {self.best_epoch + 1})")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch
        }

    def _save_checkpoint(self, epoch, train_metrics, val_metrics, is_best=False):
        """
        保存检查点

        Args:
            epoch: 当前周期
            train_metrics: 训练指标字典
            val_metrics: 验证指标字典
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config.__dict__,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()

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
            print(f"保存最佳模型到 {best_path}")

        # Delete old checkpoints to keep the number of checkpoints within max_checkpoints
        self._prune_old_checkpoints()

    def _resume_checkpoint(self, checkpoint_path=None, only_model=False):
        """
        Resume training from a checkpoint

        Args:
            checkpoint_path: Path to the checkpoint, if None, load the latest checkpoint
            only_model: Whether to load only the model weights and not the training state
        """

        if checkpoint_path is None:
            latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
            if os.path.exists(latest_path):
                checkpoint_path = latest_path
            else:
                raise ValueError("No checkpoint path specified and the latest checkpoint could not be found")

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])

        if not only_model:
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.best_metric = checkpoint.get('best_metric', float('inf'))
            self.best_epoch = checkpoint.get('best_epoch', 0)

            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.learning_rates = checkpoint.get('learning_rates', [])

            if 'optimizer_state' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            if 'scheduler_state' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])

    def _prune_old_checkpoints(self):

        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir)
                            if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]

        if len(checkpoint_files) <= self.config.max_checkpoints:
            return

        checkpoint_epochs = [(f, int(f.split("_")[-1].split(".")[0])) for f in checkpoint_files]
        checkpoint_epochs.sort(key=lambda x: x[1])

        for f, _ in checkpoint_epochs[:-self.config.max_checkpoints]:
            os.remove(os.path.join(self.checkpoint_dir, f))

    # def _plot_progress(self):
    #     """绘制训练进度"""
    #     # 检查是否有足够的数据来绘图
    #     if not self.train_losses or not self.learning_rates:
    #         print("self.train_losses:----------", self.train_losses)
    #         print("self.learning_rates:----------", self.learning_rates)
    #         print("没有足够的训练数据来绘制进度图")
    #         return
    #
    #     # 确保epochs列表与train_losses长度一致
    #     epochs = list(range(1, len(self.train_losses) + 1))
    #
    #     # 创建图表和坐标轴
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    #
    #     # 绘制损失曲线
    #     ax1.plot(epochs, self.train_losses, label='Train Loss')
    #     if self.val_losses:
    #         # 确保val_losses与epochs长度匹配
    #         val_epochs = epochs[:len(self.val_losses)]
    #         ax1.plot(val_epochs, self.val_losses, label='Val Loss')
    #     ax1.set_xlabel('Epoch')
    #     ax1.set_ylabel('Loss')
    #     ax1.set_title('Training Progress')
    #     ax1.legend()
    #     ax1.grid(True)
    #
    #     # 绘制学习率曲线
    #     # 确保learning_rates与epochs长度匹配
    #     lr_epochs = epochs[:len(self.learning_rates)]
    #     ax2.plot(lr_epochs, self.learning_rates)
    #     ax2.set_xlabel('Epoch')
    #     ax2.set_ylabel('Learning Rate')
    #     ax2.set_title('Learning Rate Schedule')
    #     ax2.grid(True)
    #
    #     # 保存图表
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.log_dir, f"training_progress_epoch_{self.current_epoch + 1}.png"))
    #     plt.close()


    def save_checkpoint(self, epoch, metrics, is_best=False):

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config.__dict__,
            'metrics': metrics
        }

        if 'instruction_avg_loss' in metrics:
            checkpoint['instruction_avg_loss'] = metrics['instruction_avg_loss']
        if 'block_length_avg_loss' in metrics:
            checkpoint['block_length_avg_loss'] = metrics['block_length_avg_loss']

        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
            torch.save(checkpoint, best_path)
            print(f"Saving the best model to {best_path}")

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, checkpoint_path)

        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)

        return checkpoint_path