import os
import torch
import glob
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
class EarlyStoppingCriterion:
    def __init__(self,
                 train_val_diff_threshold=0.005,
                 val_improvement_threshold=0.0001,
                 patience=3,
                 verbose=True):
        self.train_val_diff_threshold = train_val_diff_threshold
        self.val_improvement_threshold = val_improvement_threshold
        self.patience = patience
        self.verbose = verbose

        self.best_val_loss = float('inf')
        self.no_improvement_count = 0

        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.stopped_epoch = 0

    def __call__(self, epoch, train_loss, val_loss):

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.epochs.append(epoch)

        train_val_diff = abs(val_loss - train_loss)
        diff_condition = train_val_diff < self.train_val_diff_threshold

        if diff_condition:
            self.diff_count = getattr(self, 'diff_count', 0) + 1
        else:
            self.diff_count = 0

        if val_loss < self.best_val_loss - self.val_improvement_threshold:
            self.best_val_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        diff_stop_condition = self.diff_count >= self.patience
        improvement_stop_condition = self.no_improvement_count >= self.patience

        stop_training = diff_stop_condition or improvement_stop_condition

        if self.verbose:
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Train-Val Diff: {train_val_diff:.6f} (Threshold: {self.train_val_diff_threshold})")
            print(f"  Consecutive Small Diff Count: {self.diff_count}/{self.patience}")
            print(f"  No Improvement Count: {self.no_improvement_count}/{self.patience}")

            if diff_stop_condition:
                print("  Early stopping: Train and validation loss difference below threshold for consecutive epochs")
            if improvement_stop_condition:
                print("  Early stopping: No significant validation improvement for consecutive epochs")

        if stop_training:
            self.stopped_epoch = epoch
            if self.verbose:
                print(f"Training stopped at epoch {epoch}")

        return stop_training