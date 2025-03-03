# import os
# import time
# import torch
# import numpy as np
# from pathlib import Path
# from typing import Dict, List, Optional, Union, Tuple, Any, Callable
#
# from tqdm import tqdm
# import matplotlib.pyplot as plt
#
# class BaseTrainer:
#     """训练器基类"""
#
#     def __init__(self, model, config, experiment_dir=None):
#         """
#         初始化训练器
#
#         Args:
#             model: 要训练的模型
#             config: 配置对象
#             experiment_dir: 实验目录，如果为None则使用config中的experiment_dir
#         """
#         self.model = model
#         self.config = config
#         self.device = torch.device(config.device)
#         self.model.to(self.device)
#
#         # 训练状态
#         self.start_epoch = 0
#         self.current_epoch = 0
#         self.global_step = 0
#         self.best_metric = float('inf')  # 对于MSE等损失，越小越好
#         self.best_epoch = 0
#         self.early_stopping_counter = 0
#
#         # 训练历史
#         self.train_losses = []
#         self.val_losses = []
#         self.learning_rates = []
#
#         # 实验目录
#         self.experiment_dir = experiment_dir or config.experiment_dir
#         self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
#         self.log_dir = os.path.join(self.experiment_dir, "logs")
#         self.experiment = None
#
#         # 创建必要的目录
#         os.makedirs(self.checkpoint_dir, exist_ok=True)
#         os.makedirs(self.log_dir, exist_ok=True)
#
#         # 初始化组件
#         self.optimizer = None
#         self.scheduler = None
#         self.criterion = None
#         self.metric_fn = None
#
#         # 训练设置
#         self.clip_grad_norm = config.clip_grad_norm
#
#     def setup_optimizer(self):
#         """设置优化器，子类必须实现"""
#         raise NotImplementedError("子类必须实现setup_optimizer方法")
#
#     def setup_scheduler(self):
#         """设置学习率调度器，子类必须实现"""
#         raise NotImplementedError("子类必须实现setup_scheduler方法")
#
#     def setup_criterion(self):
#         """设置损失函数，子类必须实现"""
#         raise NotImplementedError("子类必须实现setup_criterion方法")
#
#     def train_epoch(self, train_loader) -> Dict[str, float]:
#         """
#         训练一个周期
#
#         Args:
#             train_loader: 训练数据加载器
#
#         Returns:
#             包含训练指标的字典
#         """
#         self.model.train()
#         total_loss = 0.0
#
#         # 进度条
#         progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.epochs}")
#
#         for batch_idx, batch in enumerate(progress_bar):
#             # 获取数据
#             x = batch['X'].to(self.device)
#             instruction_count = batch['instruction_count'].to(self.device)
#             y = batch['Y'].to(self.device)
#
#             # 清除梯度
#             self.optimizer.zero_grad()
#
#             # 前向传播
#             output = self.model(x, instruction_count)
#
#             # 计算损失
#             loss = self.criterion(output, y)
#
#             # 反向传播
#             loss.backward()
#
#             # 梯度裁剪
#             if self.clip_grad_norm > 0:
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
#
#             # 更新参数
#             self.optimizer.step()
#
#             # 更新总损失
#             total_loss += loss.item()
#
#             # 更新进度条
#             progress_bar.set_postfix({"loss": loss.item()})
#
#             # 更新全局步数
#             self.global_step += 1
#
#         # 计算平均损失
#         avg_loss = total_loss / len(train_loader)
#         self.train_losses.append(avg_loss)
#
#         # 记录当前学习率
#         current_lr = self.optimizer.param_groups[0]['lr']
#         self.learning_rates.append(current_lr)
#
#         return {
#             "loss": avg_loss,
#             "lr": current_lr
#         }
#
#     def validate(self, val_loader) -> Dict[str, float]:
#         """
#         在验证集上评估模型
#
#         Args:
#             val_loader: 验证数据加载器
#
#         Returns:
#             包含验证指标的字典
#         """
#         self.model.eval()
#         total_loss = 0.0
#         all_preds = []
#         all_targets = []
#
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc="Validating"):
#                 # 获取数据
#                 x = batch['X'].to(self.device)
#                 instruction_count = batch['instruction_count'].to(self.device)
#                 y = batch['Y'].to(self.device)
#
#                 # 前向传播
#                 output = self.model(x, instruction_count)
#
#                 # 计算损失
#                 loss = self.criterion(output, y)
#
#                 # 更新总损失
#                 total_loss += loss.item()
#
#                 # 收集预测值和真实值
#                 all_preds.extend(output.cpu().numpy())
#                 all_targets.extend(y.cpu().numpy())
#
#         # 计算平均损失
#         avg_loss = total_loss / len(val_loader)
#         self.val_losses.append(avg_loss)
#
#         # 计算其他指标
#         metrics = {
#             "loss": avg_loss
#         }
#
#         if self.metric_fn:
#             additional_metrics = self.metric_fn(
#                 np.array(all_targets),
#                 np.array(all_preds)
#             )
#             metrics.update(additional_metrics)
#
#         return metrics
#
#     def train(self, train_loader, val_loader, num_epochs=None, resume=False, checkpoint_path=None):
#         """
#         训练模型
#
#         Args:
#             train_loader: 训练数据加载器
#             val_loader: 验证数据加载器
#             num_epochs: 训练轮数，如果为None则使用config中的epochs
#             resume: 是否从检查点恢复训练
#             checkpoint_path: 检查点路径，如果为None且resume为True则加载最新检查点
#
#         Returns:
#             训练历史
#         """
#         # 设置训练轮数
#         num_epochs = num_epochs or self.config.epochs
#
#         # 如果需要恢复训练
#         if resume:
#             self._resume_checkpoint(checkpoint_path)
#
#         # 记录开始时间
#         start_time = time.time()
#
#         print(f"开始训练: 从Epoch {self.start_epoch + 1} 到 {num_epochs}")
#
#         for epoch in range(self.start_epoch, num_epochs):
#             self.current_epoch = epoch
#
#             # 训练一个周期
#             train_metrics = self.train_epoch(train_loader)
#
#             # 在验证集上评估
#             val_metrics = self.validate(val_loader)
#
#             # 更新学习率
#             if self.scheduler:
#                 if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                     self.scheduler.step(val_metrics["loss"])
#                 else:
#                     self.scheduler.step()
#
#             # 检查是否是最佳模型
#             is_best = val_metrics["loss"] < self.best_metric
#             if is_best:
#                 self.best_metric = val_metrics["loss"]
#                 self.best_epoch = epoch
#                 self.early_stopping_counter = 0
#             else:
#                 self.early_stopping_counter += 1
#
#             # 保存检查点
#             if (epoch + 1) % self.config.save_freq == 0 or is_best:
#                 self._save_checkpoint(epoch, train_metrics, val_metrics, is_best)
#
#             # 打印进度
#             print(f"Epoch {epoch+1}/{num_epochs} - "
#                   f"Train Loss: {train_metrics['loss']:.6f} - "
#                   f"Val Loss: {val_metrics['loss']:.6f}" +
#                   (f" - Best Val Loss: {self.best_metric:.6f} (Epoch {self.best_epoch+1})" if is_best else ""))
#
#             # 绘制训练进度
#             if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
#                 self._plot_progress()
#
#             # 早停
#             if self.early_stopping_counter >= self.config.patience:
#                 print(f"早停: 验证损失在{self.config.patience}个周期内没有改善")
#                 break
#
#         # 训练完成
#         training_time = time.time() - start_time
#         print(f"训练完成! 总时间: {training_time:.2f} 秒")
#         print(f"最佳验证损失: {self.best_metric:.6f} at Epoch {self.best_epoch+1}")
#
#         # 绘制最终训练进度
#         self._plot_progress()
#
#         # 加载最佳模型
#         best_checkpoint_path = os.path.join(self.checkpoint_dir, "model_best.pth")
#         if os.path.exists(best_checkpoint_path):
#             self._resume_checkpoint(best_checkpoint_path, only_model=True)
#             print(f"已加载最佳模型 (Epoch {self.best_epoch+1})")
#
#         return {
#             "train_losses": self.train_losses,
#             "val_losses": self.val_losses,
#             "learning_rates": self.learning_rates,
#             "best_metric": self.best_metric,
#             "best_epoch": self.best_epoch
#         }
#
#     def _save_checkpoint(self, epoch, train_metrics, val_metrics, is_best=False):
#         """
#         保存检查点
#
#         Args:
#             epoch: 当前周期
#             train_metrics: 训练指标字典
#             val_metrics: 验证指标字典
#             is_best: 是否是最佳模型
#         """
#         checkpoint = {
#             'epoch': epoch,
#             'global_step': self.global_step,
#             'model_state': self.model.state_dict(),
#             'optimizer_state': self.optimizer.state_dict(),
#             'best_metric': self.best_metric,
#             'best_epoch': self.best_epoch,
#             'train_losses': self.train_losses,
#             'val_losses': self.val_losses,
#             'learning_rates': self.learning_rates,
#             'config': self.config.__dict__,
#             'train_metrics': train_metrics,
#             'val_metrics': val_metrics
#         }
#
#         if self.scheduler:
#             checkpoint['scheduler_state'] = self.scheduler.state_dict()
#
#         # 保存常规检查点
#         checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
#         torch.save(checkpoint, checkpoint_path)
#
#         # 保存最新检查点
#         latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
#         torch.save(checkpoint, latest_path)
#
#         # 如果是最佳模型，保存一个额外的副本
#         if is_best:
#             best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
#             torch.save(checkpoint, best_path)
#             print(f"保存最佳模型到 {best_path}")
#
#         # 删除旧检查点，保持数量不超过max_checkpoints
#         self._prune_old_checkpoints()
#
#     def _resume_checkpoint(self, checkpoint_path=None, only_model=False):
#         """
#         从检查点恢复训练
#
#         Args:
#             checkpoint_path: 检查点路径，如果为None则加载最新检查点
#             only_model: 是否只加载模型权重，不恢复训练状态
#         """
#         # 如果未指定检查点，使用最新检查点
#         if checkpoint_path is None:
#             latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
#             if os.path.exists(latest_path):
#                 checkpoint_path = latest_path
#             else:
#                 raise ValueError("未指定检查点路径且找不到最新检查点")
#
#         print(f"加载检查点: {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
#
#         # 加载模型状态
#         self.model.load_state_dict(checkpoint['model_state'])
#
#         if not only_model:
#             # 恢复训练状态
#             self.start_epoch = checkpoint['epoch'] + 1
#             self.global_step = checkpoint.get('global_step', 0)
#             self.best_metric = checkpoint.get('best_metric', float('inf'))
#             self.best_epoch = checkpoint.get('best_epoch', 0)
#
#             # 恢复历史数据
#             self.train_losses = checkpoint.get('train_losses', [])
#             self.val_losses = checkpoint.get('val_losses', [])
#             self.learning_rates = checkpoint.get('learning_rates', [])
#
#             # 恢复优化器状态
#             if 'optimizer_state' in checkpoint and self.optimizer:
#                 self.optimizer.load_state_dict(checkpoint['optimizer_state'])
#
#             # 恢复调度器状态
#             if 'scheduler_state' in checkpoint and self.scheduler:
#                 self.scheduler.load_state_dict(checkpoint['scheduler_state'])
#
#     def _prune_old_checkpoints(self):
#         """删除旧检查点，只保留最近的几个和最佳模型"""
#         # 获取所有检查点文件
#         checkpoint_files = [f for f in os.listdir(self.checkpoint_dir)
#                           if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
#
#         # 如果检查点数量不超过限制，不做任何操作
#         if len(checkpoint_files) <= self.config.max_checkpoints:
#             return
#
#         # 解析周期编号
#         checkpoint_epochs = [(f, int(f.split("_")[-1].split(".")[0])) for f in checkpoint_files]
#
#         # 按周期排序
#         checkpoint_epochs.sort(key=lambda x: x[1])
#
#         # 删除最旧的检查点，直到数量符合限制
#         for f, _ in checkpoint_epochs[:-self.config.max_checkpoints]:
#             os.remove(os.path.join(self.checkpoint_dir, f))
#
#     def _plot_progress(self):
#         """绘制训练进度"""
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#
#         # 绘制损失曲线
#         epochs = list(range(1, len(self.train_losses) + 1))
#         ax1.plot(epochs, self.train_losses, label='Train Loss')
#         ax1.plot(epochs, self.val_losses, label='Val Loss')
#         ax1.set_xlabel('Epoch')
#         ax1.set_ylabel('Loss')
#         ax1.set_title('Training Progress')
#         ax1.legend()
#         ax1.grid(True)
#
#         # 绘制学习率曲线
#         ax2.plot(epochs, self.learning_rates)
#         ax2.set_xlabel('Epoch')
#         ax2.set_ylabel('Learning Rate')
#         ax2.set_title('Learning Rate Schedule')
#         ax2.grid(True)
#
#         # 保存图表
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.log_dir, f"training_progress_epoch_{self.current_epoch+1}.png"))
#         plt.close()
#
#     def save_checkpoint(self, epoch, metrics, is_best=False):
#         """保存检查点"""
#         checkpoint = {
#             'epoch': epoch,
#             'global_step': self.global_step,
#             'model_state': self.model.state_dict(),
#             'optimizer_state': self.optimizer.state_dict(),
#             'best_metric': self.best_metric,
#             'best_epoch': self.best_epoch,
#             'train_losses': self.train_losses,
#             'val_losses': self.val_losses,
#             'learning_rates': self.learning_rates,
#             'config': self.config.__dict__,
#             'metrics': metrics
#         }
#
#         # 保存特殊指标统计
#         if 'instruction_avg_loss' in metrics:
#             checkpoint['instruction_avg_loss'] = metrics['instruction_avg_loss']
#         if 'block_length_avg_loss' in metrics:
#             checkpoint['block_length_avg_loss'] = metrics['block_length_avg_loss']
#
#         if self.scheduler:
#             checkpoint['scheduler_state'] = self.scheduler.state_dict()
#
#         # 如果是最佳模型，保存到特定文件
#         if is_best:
#             best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
#             torch.save(checkpoint, best_path)
#             print(f"保存最佳模型到 {best_path}")
#
#         # 保存常规检查点
#         checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
#         torch.save(checkpoint, checkpoint_path)
#
#         # 保存最新检查点
#         latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
#         torch.save(checkpoint, latest_path)
#
#         return checkpoint_path

import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from tqdm import tqdm
import matplotlib.pyplot as plt


class BaseTrainer:
    """训练器基类"""

    def __init__(self, model, config, experiment_dir=None):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            config: 配置对象
            experiment_dir: 实验目录，如果为None则使用config中的experiment_dir
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # 训练状态
        self.start_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')  # 对于MSE等损失，越小越好
        self.best_epoch = 0
        self.early_stopping_counter = 0

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # 实验目录
        self.experiment_dir = experiment_dir or config.experiment_dir
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir = os.path.join(self.experiment_dir, "logs")

        # 创建必要的目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化组件
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metric_fn = None

        # 训练设置
        self.clip_grad_norm = config.clip_grad_norm

        # 初始化experiment属性，默认为None
        self.experiment = None

    def setup_experiment(self, experiment_name=None, base_dir=None):
        """
        设置实验管理器

        Args:
            experiment_name: 实验名称，如果为None则使用config中的experiment_name或'unnamed_experiment'
            base_dir: 实验基础目录，如果为None则使用config中的experiment_base_dir或'experiments'
        """
        from utils.experiment import ExperimentManager

        experiment_name = experiment_name or getattr(self.config, 'experiment_name', 'unnamed_experiment')
        base_dir = base_dir or getattr(self.config, 'experiment_base_dir', 'experiments')

        self.experiment = ExperimentManager(experiment_name, base_dir)
        # 保存配置
        self.experiment.save_config(self.config)

        return self.experiment

    def setup_optimizer(self):
        """设置优化器，子类必须实现"""
        raise NotImplementedError("子类必须实现setup_optimizer方法")

    def setup_scheduler(self):
        """设置学习率调度器，子类必须实现"""
        raise NotImplementedError("子类必须实现setup_scheduler方法")

    def setup_criterion(self):
        """设置损失函数，子类必须实现"""
        raise NotImplementedError("子类必须实现setup_criterion方法")

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
            train_metrics = self.train_epoch(train_loader)

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

            # 打印进度
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_metrics['loss']:.6f} - "
                  f"Val Loss: {val_metrics['loss']:.6f}" +
                  (f" - Best Val Loss: {self.best_metric:.6f} (Epoch {self.best_epoch + 1})" if is_best else ""))

            # 绘制训练进度
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self._plot_progress()

            # 早停
            if self.early_stopping_counter >= self.config.patience:
                print(f"早停: 验证损失在{self.config.patience}个周期内没有改善")
                break

        # 训练完成
        training_time = time.time() - start_time
        print(f"训练完成! 总时间: {training_time:.2f} 秒")
        print(f"最佳验证损失: {self.best_metric:.6f} at Epoch {self.best_epoch + 1}")

        # 绘制最终训练进度
        self._plot_progress()

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

        # 删除旧检查点，保持数量不超过max_checkpoints
        self._prune_old_checkpoints()

    def _resume_checkpoint(self, checkpoint_path=None, only_model=False):
        """
        从检查点恢复训练

        Args:
            checkpoint_path: 检查点路径，如果为None则加载最新检查点
            only_model: 是否只加载模型权重，不恢复训练状态
        """
        # 如果未指定检查点，使用最新检查点
        if checkpoint_path is None:
            latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
            if os.path.exists(latest_path):
                checkpoint_path = latest_path
            else:
                raise ValueError("未指定检查点路径且找不到最新检查点")

        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state'])

        if not only_model:
            # 恢复训练状态
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.best_metric = checkpoint.get('best_metric', float('inf'))
            self.best_epoch = checkpoint.get('best_epoch', 0)

            # 恢复历史数据
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.learning_rates = checkpoint.get('learning_rates', [])

            # 恢复优化器状态
            if 'optimizer_state' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            # 恢复调度器状态
            if 'scheduler_state' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])

    def _prune_old_checkpoints(self):
        """删除旧检查点，只保留最近的几个和最佳模型"""
        # 获取所有检查点文件
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir)
                            if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]

        # 如果检查点数量不超过限制，不做任何操作
        if len(checkpoint_files) <= self.config.max_checkpoints:
            return

        # 解析周期编号
        checkpoint_epochs = [(f, int(f.split("_")[-1].split(".")[0])) for f in checkpoint_files]

        # 按周期排序
        checkpoint_epochs.sort(key=lambda x: x[1])

        # 删除最旧的检查点，直到数量符合限制
        for f, _ in checkpoint_epochs[:-self.config.max_checkpoints]:
            os.remove(os.path.join(self.checkpoint_dir, f))

    def _plot_progress(self):
        """绘制训练进度"""
        # 检查是否有足够的数据来绘图
        if not self.train_losses or not self.learning_rates:
            print("self.train_losses:----------", self.train_losses)
            print("self.learning_rates:----------", self.learning_rates)
            print("没有足够的训练数据来绘制进度图")
            return

        # 确保epochs列表与train_losses长度一致
        epochs = list(range(1, len(self.train_losses) + 1))

        # 创建图表和坐标轴
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 绘制损失曲线
        ax1.plot(epochs, self.train_losses, label='Train Loss')
        if self.val_losses:
            # 确保val_losses与epochs长度匹配
            val_epochs = epochs[:len(self.val_losses)]
            ax1.plot(val_epochs, self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)

        # 绘制学习率曲线
        # 确保learning_rates与epochs长度匹配
        lr_epochs = epochs[:len(self.learning_rates)]
        ax2.plot(lr_epochs, self.learning_rates)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)

        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"training_progress_epoch_{self.current_epoch + 1}.png"))
        plt.close()


    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
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

        # 保存特殊指标统计
        if 'instruction_avg_loss' in metrics:
            checkpoint['instruction_avg_loss'] = metrics['instruction_avg_loss']
        if 'block_length_avg_loss' in metrics:
            checkpoint['block_length_avg_loss'] = metrics['block_length_avg_loss']

        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()

        # 如果是最佳模型，保存到特定文件
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到 {best_path}")

        # 保存常规检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, checkpoint_path)

        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)

        return checkpoint_path