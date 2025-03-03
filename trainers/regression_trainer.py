import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

import numpy as np
from typing import Dict, Callable, List, Optional, Union
from tqdm import tqdm

from .base_trainer import BaseTrainer
from utils.metrics import compute_regression_metrics, MapeLoss, BatchResult, correct_regression


class RegressionTrainer(BaseTrainer):
    """回归任务训练器"""

    def __init__(self, model, config, experiment_dir=None):
        """
        初始化回归训练器

        Args:
            model: 要训练的模型
            config: 配置对象
            experiment_dir: 实验目录
        """
        super(RegressionTrainer, self).__init__(model, config, experiment_dir)

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
            包含训练指标的字典
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

        return metrics

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
        if hasattr(self, 'experiment') and self.experiment is not None:
            self.experiment.log_metrics(metrics, epoch, prefix="val_")

        return metrics