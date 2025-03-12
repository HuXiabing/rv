import os
import time
import torch
import numpy as np
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

    # def train_epoch(self, train_loader) -> Dict[str, float]:
    #
    #     self.model.train()
    #     total_loss = 0.0
    #
    #     progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.epochs}")
    #
    #     for batch_idx, batch in enumerate(progress_bar):
    #
    #         x = batch['X'].to(self.device)
    #         y = batch['Y'].to(self.device)
    #         instruction_count = batch.get('instruction_count', None)
    #
    #         self.optimizer.zero_grad()
    #
    #         output = self.model(x)
    #         loss = self.criterion(output, y)
    #         mean_loss = torch.mean(loss)
    #         mean_loss.backward()
    #
    #         if self.clip_grad_norm > 0:
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
    #
    #         self.optimizer.step()
    #
    #         total_loss += mean_loss.item()
    #         progress_bar.set_postfix({"loss": mean_loss.item()})
    #         self.global_step += 1
    #
    #     # avg_loss = total_loss / len(train_loader)
    #     self.train_losses.append(mean_loss)
    #
    #     current_lr = self.optimizer.param_groups[0]['lr']
    #     self.learning_rates.append(current_lr)
    #
    #     return {
    #         "loss": mean_loss,
    #         "lr": current_lr
    #     }
    #
    # def validate(self, val_loader) -> Dict[str, float]:
    #
    #     self.model.eval()
    #     total_loss = 0.0
    #     all_preds = []
    #     all_targets = []
    #
    #     with torch.no_grad():
    #         for batch in tqdm(val_loader, desc="Validating"):
    #             x = batch['X'].to(self.device)
    #             y = batch['Y'].to(self.device)
    #             instruction_count = batch.get('instruction_count', None)
    #
    #             output = self.model(x)
    #             loss = self.criterion(output, y)
    #             total_loss += loss.item()
    #
    #             all_preds.extend(output.cpu().numpy())
    #             all_targets.extend(y.cpu().numpy())
    #
    #     avg_loss = total_loss / len(val_loader)
    #     self.val_losses.append(avg_loss)
    #
    #     metrics = {
    #         "loss": avg_loss
    #     }
    #
    #     if self.metric_fn:
    #         additional_metrics = self.metric_fn(
    #             np.array(all_targets),
    #             np.array(all_preds)
    #         )
    #         metrics.update(additional_metrics)
    #
    #     return metrics
    #
    # def train(self, train_loader, val_loader, num_epochs=None, resume=False, checkpoint_path=None):
    #
    #     num_epochs = num_epochs or self.config.epochs
    #
    #     if resume:
    #         self._resume_checkpoint(checkpoint_path)
    #
    #     start_time = time.time()
    #
    #     print(f"Starting training---------------------------\nFrom epoch {self.start_epoch + 1} to {num_epochs}")
    #
    #     for epoch in range(self.start_epoch, num_epochs):
    #         self.current_epoch = epoch
    #
    #         train_metrics, train_batch_result = self.train_epoch(train_loader)
    #
    #         val_metrics = self.validate(val_loader)
    #
    #         if self.scheduler:
    #             if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #                 self.scheduler.step(val_metrics["loss"])
    #             else:
    #                 self.scheduler.step()
    #
    #         is_best = val_metrics["loss"] < self.best_metric
    #         if is_best:
    #             self.best_metric = val_metrics["loss"]
    #             self.best_epoch = epoch
    #             self.early_stopping_counter = 0
    #         else:
    #             self.early_stopping_counter += 1
    #
    #         if (epoch + 1) % self.config.save_freq == 0 or is_best:
    #             self._save_checkpoint(epoch, train_metrics, val_metrics, is_best)
    #
    #         instruction_stats = {
    #                 "instruction_avg_loss": train_batch_result.get_instruction_avg_loss(),
    #                 "instruction_counts": train_batch_result.instruction_counts
    #         }
    #
    #         block_length_stats = {
    #             "block_length_avg_loss": train_batch_result.get_block_length_avg_loss(),
    #             "block_length_counts": train_batch_result.block_lengths_counts
    #         }
    #
    #         if hasattr(self, 'experiment') and self.experiment:
    #             self.experiment.save_instruction_stats(instruction_stats, epoch)
    #             self.experiment.save_block_length_stats(block_length_stats, epoch)
    #
    #             self.experiment.visualize_epoch_stats(
    #                 instruction_stats,
    #                 block_length_stats,
    #                 epoch
    #             )
    #
    #         print(f"Epoch {epoch + 1}/{num_epochs} - "
    #               f"Train Loss: {train_metrics['loss']:.6f} - "
    #               f"Val Loss: {val_metrics['loss']:.6f}" +
    #               (f" - Best Val Loss: {self.best_metric:.6f} (Epoch {self.best_epoch + 1})" if is_best else ""))
    #
    #         if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
    #             self._plot_progress()
    #
    #         if self.early_stopping_counter >= self.config.patience:
    #             print(f"Early stopping: Validation loss did not improve for {self.config.patience} epochs")
    #             break
    #
    #     training_time = time.time() - start_time
    #     print(f"Training completed! Total time: {training_time:.2f} seconds")
    #     print(f"Best validation loss: {self.best_metric:.6f} at Epoch {self.best_epoch + 1}")
    #
    #     self._plot_progress()
    #
    #     return {
    #         "train_losses": self.train_losses,
    #         "val_losses": self.val_losses,
    #         "learning_rates": self.learning_rates,
    #         "best_metric": self.best_metric,
    #         "best_epoch": self.best_epoch
    #     }

    def _save_checkpoint(self, epoch, train_metrics=None, val_metrics=None, is_best=False):

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

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, checkpoint_path)

        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
            torch.save(checkpoint, best_path)
            print(f"Saving the best model to {best_path}")

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

    def _plot_progress(self):
        """
        plot the training progress
        """
        epochs = list(range(1, len(self.train_losses) + 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, self.train_losses, label='Train Loss')
        if self.val_losses:
            val_epochs = epochs[:len(self.val_losses)]
            ax1.plot(val_epochs, self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)

        lr_epochs = epochs[:len(self.learning_rates)]
        ax2.plot(lr_epochs, self.learning_rates)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"training_progress_epoch_{self.current_epoch + 1}.png"))
        plt.close()

