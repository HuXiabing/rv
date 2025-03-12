import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn

class MapeLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none')
        self.epsilon = epsilon

    def forward(self, output, target):
        loss = self.loss(output, target) / (torch.abs(target) + self.epsilon)
        return loss

class BatchResult:

    def __init__(self):
        self.batch_len = 0
        self.measured = []
        self.prediction = []
        self.inst_lens = []
        self.index = []
        self.loss_sum = 0

        self.instruction_losses = {}
        self.block_lengths_losses = {}
        self.instruction_counts = {}
        self.block_lengths_counts = {}

    @property
    def loss(self):
        if self.batch_len == 0:
            return float('nan')
        # return self.loss_sum / self.batch_len
        return self.loss_sum / len(self.prediction)

    def __iadd__(self, other):

        self.batch_len += other.batch_len

        self.measured.extend(other.measured)
        self.prediction.extend(other.prediction)
        self.inst_lens.extend(other.inst_lens)
        self.index.extend(other.index)
        self.loss_sum += other.loss_sum

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

    def add_sample(self, prediction, measured, loss, instructions=None, block_len=None):

        self.batch_len += 1
        self.prediction.append(prediction)
        self.measured.append(measured)
        self.loss_sum += loss

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

    def get_instruction_avg_loss(self):
        """Get the average loss for each instruction type"""
        return {instr_type: loss / count
                for instr_type, loss in self.instruction_losses.items()
                for count in [self.instruction_counts.get(instr_type, 1)]
                if count > 0}

    def get_block_length_avg_loss(self):
        """Get the average loss for each basic block length"""
        return {block_len: loss / count
                for block_len, loss in self.block_lengths_losses.items()
                for count in [self.block_lengths_counts.get(block_len, 1)]
                if count > 0}

    def compute_metrics(self, tolerances=25):
        y_true = np.array(self.measured)
        y_pred = np.array(self.prediction)

        metrics = compute_regression_metrics(y_true, y_pred, tolerances)
        metrics["loss"] = self.loss

        metrics["instruction_avg_loss"] = self.get_instruction_avg_loss()
        metrics["block_length_avg_loss"] = self.get_block_length_avg_loss()
        metrics["instruction_counts"] = self.instruction_counts
        metrics["block_length_counts"] = self.block_lengths_counts

        # return metrics
        return {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"]
        }

def correct_regression(pred, answer, tolerance=25):
    """
    Calculate the correctness rate of regression predictions

    Args:
        pred: Predicted values
        answer: True values
        tolerance: Tolerance percentage, default 10%

    Returns:
        Number of correct predictions
    """
    if isinstance(pred, list):
        pred = torch.tensor(pred)
    if isinstance(answer, list):
        answer = torch.tensor(answer)

    percentage = torch.abs(pred - answer) * 100.0 / (torch.abs(answer) + 1e-3)
    return torch.sum(percentage < tolerance).item()

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tolerance=25) -> float:
    """
    Calculate the accuracy of predictions

    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        tolerance: Tolerance percentage, default 10%

    Returns:
        Accuracy (float between 0 and 1)
    """
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    correct_count = correct_regression(y_pred_tensor, y_true_tensor, tolerance)
    total_count = len(y_true)

    return correct_count / total_count if total_count > 0 else 0.0

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, tolerance=25) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics

    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        tolerance: List of different tolerance levels for calculating accuracy under different criteria

    Returns:
        Dictionary containing evaluation metrics
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    #MPE
    mpe = np.mean((y_true - y_pred) / (np.abs(y_true) + epsilon)) * 100
    accuracy = compute_accuracy(y_true, y_pred, tolerance)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "mpe": mpe,
        "accuracy": accuracy
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

