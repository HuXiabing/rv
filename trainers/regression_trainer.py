import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

import numpy as np
from typing import Dict, Callable, List, Optional, Union
from tqdm import tqdm

from .base_trainer import BaseTrainer
from utils.metrics import compute_regression_metrics, MapeLoss, BatchResult, correct_regression
import time
import os

class RegressionTrainer(BaseTrainer):
    """回归任务训练器"""

    def __init__(self, model, config, experiment_dir=None, experiment=None):
        """
        初始化回归训练器

        Args:
            model: 要训练的模型
            config: 配置对象
            experiment_dir: 实验目录
        """
        super(RegressionTrainer, self).__init__(model, config, experiment_dir)
        self.experiment = experiment

        # 设置训练组件
        self.setup_criterion()
        self.setup_optimizer()
        self.setup_scheduler()

        # 统计信息和最佳结果
        self.best_accuracy = 0.0
        self.accuracy_tolerance = getattr(config, 'accuracy_tolerance', 10.0)

        # 设置评估指标
        self.metric_fn = lambda y_true, y_pred: compute_regression_metrics(
            y_true, y_pred, [5.0, 10.0, 15.0]
        )

    def setup_criterion(self):
        """设置损失函数"""
        loss_type = getattr(self.config, 'loss_type', 'mape').lower()

        if loss_type == 'mape':
            # 使用MAPE损失函数
            epsilon = getattr(self.config, 'loss_epsilon', 1e-5)
            self.criterion = MapeLoss(epsilon=epsilon)
        elif loss_type == 'mae' or loss_type == 'l1':
            # 使用MAE/L1损失函数
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'huber':
            # 使用Huber损失函数
            delta = getattr(self.config, 'huber_delta', 1.0)
            self.criterion = nn.HuberLoss(delta=delta, reduction='none')
        else:
            # 默认使用MSE损失函数
            self.criterion = nn.MSELoss(reduction='none')

    def setup_optimizer(self):
        """设置优化器"""
        # 根据配置选择不同的优化器
        optimizer_name = getattr(self.config, 'optimizer', 'adam').lower()

        if optimizer_name == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        else:  # 默认使用Adam
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

    def setup_scheduler(self):
        """设置学习率调度器"""
        # 根据配置选择不同的调度器
        scheduler_name = getattr(self.config, 'scheduler', 'plateau').lower()

        if scheduler_name == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.lr / 100
            )
        elif scheduler_name == 'step':
            step_size = getattr(self.config, 'step_size', 10)
            gamma = getattr(self.config, 'gamma', 0.1)
            self.scheduler = StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        else:  # 默认使用ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.patience // 2,
                verbose=True
            )

    def train_epoch(self, train_loader):
        """
        训练一个周期

        Args:
            train_loader: 训练数据加载器

        Returns:
            包含训练指标的字典和BatchResult对象
        """
        self.model.train()
        batch_result = BatchResult()

        # 进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.epochs}")

        for batch in progress_bar:
            # 获取数据
            x = batch['X'].to(self.device)
            instruction_count = batch.get('instruction_count', None)
            # if instruction_count is not None:
            #     instruction_count = instruction_count.to(self.device)
            y = batch['Y'].to(self.device)

            # 清除梯度
            self.optimizer.zero_grad()

            # 前向传播
            # output = self.model(x, instruction_count)
            output = self.model(x)

            # 计算损失
            loss = self.criterion(output, y)
            mean_loss = torch.mean(loss)

            # 反向传播
            mean_loss.backward()

            # 梯度裁剪
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            # 更新参数
            self.optimizer.step()

            # 收集批次统计
            for i in range(len(output)):
                # 提取指令类型信息
                instructions = []
                block_len = None

                # 如果有效指令数量可用，则提取基本块长度
                if instruction_count is not None:
                    valid_count = instruction_count[i].item()
                    block_len = valid_count

                    # 提取指令类型
                    for j in range(valid_count):
                        # 获取第一个非零标记，通常是指令类型
                        instr_tokens = [t.item() for t in x[i, j] if t.item() != 0]
                        if instr_tokens:
                            instructions.append(instr_tokens[0])  # 假设第一个令牌是指令类型

                # 添加样本结果
                batch_result.add_sample(
                    prediction=output[i].item(),
                    measured=y[i].item(),
                    loss=loss[i].item(),
                    instructions=instructions,
                    block_len=block_len
                )

            # 更新进度条
            progress_bar.set_postfix({"loss": mean_loss.item()})

            # 更新全局步数
            self.global_step += 1

        # 计算训练指标
        metrics = batch_result.compute_metrics([self.accuracy_tolerance])

        # 记录当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        metrics["lr"] = current_lr
        self.learning_rates.append(current_lr)

        # 打印详细统计信息
        print(f"\nTraining Statistics - Epoch {self.current_epoch + 1}:")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  Accuracy ({self.accuracy_tolerance}%): {metrics.get(f'accuracy_{self.accuracy_tolerance:.1f}', 0):.6f}")

        # 记录详细信息到实验管理器，如果存在
        if hasattr(self, 'experiment') and self.experiment is not None:
            self.experiment.log_metrics(metrics, self.current_epoch, prefix="train_")

        return metrics, batch_result

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

        print(f"Starting training: from epoch {self.start_epoch + 1} to {num_epochs}")

        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch

            train_metrics, train_batch_result = self.train_epoch(train_loader)

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

    def validate(self, val_loader, epoch=None):
        """
        在验证集上评估模型

        Args:
            val_loader: 验证数据加载器
            epoch: 当前周期，如果为None则使用self.current_epoch

        Returns:
            包含验证指标的字典
        """
        # 如果没有提供epoch参数，使用当前epoch
        if epoch is None:
            epoch = self.current_epoch

        self.model.eval()
        batch_result = BatchResult()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # 获取数据
                x = batch['X'].to(self.device)
                instruction_count = batch.get('instruction_count', None)
                # if instruction_count is not None:
                #     instruction_count = instruction_count.to(self.device)
                y = batch['Y'].to(self.device)

                # 前向传播
                # output = self.model(x, instruction_count)
                output = self.model(x)

                # 计算损失
                loss = self.criterion(output, y)

                # 收集批次统计
                for i in range(len(output)):
                    # 提取指令类型信息
                    instructions = []
                    block_len = None

                    # 如果有效指令数量可用，则提取基本块长度
                    if instruction_count is not None:
                        valid_count = instruction_count[i].item()
                        block_len = valid_count

                        # 提取指令类型
                        for j in range(valid_count):
                            # 获取第一个非零标记，通常是指令类型
                            instr_tokens = [t.item() for t in x[i, j] if t.item() != 0]
                            if instr_tokens:
                                instructions.append(instr_tokens[0])  # 假设第一个令牌是指令类型

                    # 添加样本结果
                    batch_result.add_sample(
                        prediction=output[i].item(),
                        measured=y[i].item(),
                        loss=loss[i].item(),
                        instructions=instructions,
                        block_len=block_len
                    )

        # 计算所有验证指标
        metrics = batch_result.compute_metrics([self.accuracy_tolerance])

        # 打印详细统计信息
        accuracy_key = f"accuracy_{self.accuracy_tolerance:.1f}"
        current_accuracy = metrics.get(accuracy_key, 0)

        print(f"\nValidation Results - Epoch {epoch + 1}:")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  Accuracy ({self.accuracy_tolerance}%): {current_accuracy:.6f}")

        # 检查是否是最佳准确率
        is_best_accuracy = current_accuracy > self.best_accuracy
        if is_best_accuracy:
            self.best_accuracy = current_accuracy
            metrics["is_best_accuracy"] = True

        # 记录详细信息到实验管理器，如果存在
        # if hasattr(self, 'experiment') and self.experiment is not None:
        self.experiment.log_metrics(metrics, epoch, prefix="val_")

        return metrics