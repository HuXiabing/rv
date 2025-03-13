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
    def loss(self):  # average loss of each sample
        if self.batch_len == 0:
            return float('nan')
        return self.loss_sum / self.batch_len

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
        print("y_true", len(y_pred))

        metrics = {}

        metrics["accuracy"] = compute_accuracy(y_true, y_pred, tolerances)
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
        pred: Predicted values --> tensor
        answer: True values --> tensor
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
        y_true:
        y_pred:
        tolerance: Tolerance percentage, default 10%

    Returns:
        Accuracy (float between 0 and 1)
    """

    correct_count = correct_regression(torch.tensor(y_pred), torch.tensor(y_true), tolerance)
    total_count = len(y_true)

    return correct_count / total_count if total_count > 0 else 0.0

# def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, tolerance=25) -> Dict[str, float]:
#     """
#     Calculate regression evaluation metrics
#
#     Args:
#         y_true: Array of true values
#         y_pred: Array of predicted values
#         tolerance: List of different tolerance levels for calculating accuracy under different criteria
#
#     Returns:
#         Dictionary containing evaluation metrics
#     """
#
#     mse = mean_squared_error(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     # MAPE
#     epsilon = 1e-10
#     mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
#     #MPE
#     mpe = np.mean((y_true - y_pred) / (np.abs(y_true) + epsilon)) * 100
#     accuracy = compute_accuracy(y_true, y_pred, tolerance)
#
#     metrics = {
#         "mse": mse,
#         "mae": mae,
#         "r2": r2,
#         "mape": mape,
#         "mpe": mpe,
#         "accuracy": accuracy
#     }
#
#     return metrics

