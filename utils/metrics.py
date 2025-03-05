import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn

class MapeLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction='none')
        self.epsilon = epsilon

    def forward(self, output, target):
        loss = self.loss_fn(output, target) / (torch.abs(target) + self.epsilon)
        return loss

def correct_regression(pred, answer, tolerance=10.0):
    """
    计算回归预测的正确率

    Args:
        pred: 预测值
        answer: 真实值
        tolerance: 容忍度百分比，默认10%

    Returns:
        正确预测的数量
    """
    if isinstance(pred, list):
        pred = torch.tensor(pred)
    if isinstance(answer, list):
        answer = torch.tensor(answer)

    percentage = torch.abs(pred - answer) * 100.0 / (torch.abs(answer) + 1e-3)
    return torch.sum(percentage < tolerance).item()


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tolerance=10.0) -> float:
    """
    计算预测的准确率

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        tolerance: 容忍度百分比，默认10%

    Returns:
        准确率 (0-1之间的浮点数)
    """
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    correct_count = correct_regression(y_pred_tensor, y_true_tensor, tolerance)
    total_count = len(y_true)

    return correct_count / total_count if total_count > 0 else 0.0


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, tolerances=[5.0, 10.0, 15.0]) -> Dict[
    str, float]:
    """
    计算回归评估指标

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        tolerances: 不同容忍度的列表，用于计算不同标准下的准确率

    Returns:
        包含评估指标的字典
    """
    # 计算均方误差(MSE)
    mse = mean_squared_error(y_true, y_pred)

    # 计算均方根误差(RMSE)
    rmse = np.sqrt(mse)

    # 计算平均绝对误差(MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # 计算决定系数(R²)
    r2 = r2_score(y_true, y_pred)

    # 计算平均绝对百分比误差(MAPE)
    epsilon = 1e-10  # 防止除零错误
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

    # 计算平均百分比误差(MPE)
    mpe = np.mean((y_true - y_pred) / (np.abs(y_true) + epsilon)) * 100

    # 计算不同容忍度下的准确率
    accuracy_metrics = {}
    for tolerance in tolerances:
        acc = compute_accuracy(y_true, y_pred, tolerance)
        accuracy_metrics[f"accuracy_{tolerance:.1f}"] = acc

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "mpe": mpe,
        **accuracy_metrics
    }

    return metrics

def compute_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    计算预测误差分布

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组

    Returns:
        包含误差分布的字典
    """
    # 计算误差
    errors = y_true - y_pred

    # 计算相对误差
    epsilon = 1e-10  # 防止除零错误
    rel_errors = errors / (np.abs(y_true) + epsilon)

    # 计算误差统计信息
    error_stats = {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "p25": float(np.percentile(errors, 25)),
        "p50": float(np.percentile(errors, 50)),
        "p75": float(np.percentile(errors, 75)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99))
    }

    # 计算相对误差统计信息
    rel_error_stats = {
        "mean_rel": float(np.mean(rel_errors) * 100),  # 转为百分比
        "std_rel": float(np.std(rel_errors) * 100),
        "min_rel": float(np.min(rel_errors) * 100),
        "max_rel": float(np.max(rel_errors) * 100),
        "p25_rel": float(np.percentile(rel_errors, 25) * 100),
        "p50_rel": float(np.percentile(rel_errors, 50) * 100),
        "p75_rel": float(np.percentile(rel_errors, 75) * 100),
        "p90_rel": float(np.percentile(rel_errors, 90) * 100),
        "p95_rel": float(np.percentile(rel_errors, 95) * 100),
        "p99_rel": float(np.percentile(rel_errors, 99) * 100)
    }

    return {
        "errors": errors,
        "rel_errors": rel_errors * 100,  # 转为百分比
        "error_stats": error_stats,
        "rel_error_stats": rel_error_stats
    }

class BatchResult:
    """
    保存批次训练或验证的结果
    """

    def __init__(self):
        self.batch_len = 0

        self.measured = []  # 真实值
        self.prediction = []  # 预测值
        self.inst_lens = []  # 每个样本的指令长度
        self.index = []  # 样本索引

        self.loss_sum = 0

        # 详细的统计数据
        self.instruction_losses = {}  # 每种指令类型的损失总和
        self.block_lengths_losses = {}  # 每种基本块长度的损失总和
        self.instruction_counts = {}  # 每种指令类型的出现次数
        self.block_lengths_counts = {}  # 每种基本块长度的出现次数

    @property
    def loss(self):
        if self.batch_len == 0:
            return float('nan')
        return self.loss_sum / self.batch_len

    def add_sample(self, prediction, measured, loss, instructions=None, block_len=None):
        """
        添加单个样本的结果

        Args:
            prediction: 预测值
            measured: 真实值
            loss: 损失值
            instructions: 指令类型列表
            block_len: 基本块长度
        """
        self.batch_len += 1
        self.prediction.append(prediction)
        self.measured.append(measured)
        self.loss_sum += loss

        # 更新指令类型和基本块长度的统计
        if instructions is not None:
            for instr_type in instructions:
                if instr_type not in self.instruction_losses:
                    self.instruction_losses[instr_type] = 0
                    self.instruction_counts[instr_type] = 0
                self.instruction_losses[instr_type] += loss / len(instructions)
                self.instruction_counts[instr_type] += 1

        if block_len is not None:
            if block_len not in self.block_lengths_losses:
                self.block_lengths_losses[block_len] = 0
                self.block_lengths_counts[block_len] = 0
            self.block_lengths_losses[block_len] += loss
            self.block_lengths_counts[block_len] += 1

    def __iadd__(self, other):
        """合并两个BatchResult对象"""
        self.batch_len += other.batch_len

        self.measured.extend(other.measured)
        self.prediction.extend(other.prediction)
        self.inst_lens.extend(other.inst_lens)
        self.index.extend(other.index)

        self.loss_sum += other.loss_sum

        # 合并指令类型和基本块长度的统计
        for instr_type, loss in other.instruction_losses.items():
            if instr_type not in self.instruction_losses:
                self.instruction_losses[instr_type] = 0
                self.instruction_counts[instr_type] = 0
            self.instruction_losses[instr_type] += loss
            self.instruction_counts[instr_type] += other.instruction_counts.get(instr_type, 0)

        for block_len, loss in other.block_lengths_losses.items():
            if block_len not in self.block_lengths_losses:
                self.block_lengths_losses[block_len] = 0
                self.block_lengths_counts[block_len] = 0
            self.block_lengths_losses[block_len] += loss
            self.block_lengths_counts[block_len] += other.block_lengths_counts.get(block_len, 0)

        return self

    def get_instruction_avg_loss(self):
        """获取每种指令类型的平均损失"""
        return {instr_type: loss / count
                for instr_type, loss in self.instruction_losses.items()
                for count in [self.instruction_counts.get(instr_type, 1)]
                if count > 0}

    def get_block_length_avg_loss(self):
        """获取每种基本块长度的平均损失"""
        return {block_len: loss / count
                for block_len, loss in self.block_lengths_losses.items()
                for count in [self.block_lengths_counts.get(block_len, 1)]
                if count > 0}

    def compute_metrics(self, tolerances=[5.0, 10.0, 15.0]):
        """计算所有评估指标"""
        y_true = np.array(self.measured)
        y_pred = np.array(self.prediction)

        metrics = compute_regression_metrics(y_true, y_pred, tolerances)
        metrics["loss"] = self.loss

        # 添加指令类型和基本块长度的统计
        metrics["instruction_avg_loss"] = self.get_instruction_avg_loss()
        metrics["block_length_avg_loss"] = self.get_block_length_avg_loss()
        metrics["instruction_counts"] = self.instruction_counts
        metrics["block_length_counts"] = self.block_lengths_counts

        return metrics