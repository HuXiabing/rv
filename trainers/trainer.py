import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from tqdm import tqdm
from data import RISCVGraphDataset
from utils.metrics import MapeLoss, BatchResult, correct_regression, compute_accuracy #,compute_regression_metrics
import time
import os
from utils.experiment import ExperimentManager
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, config, experiment_dir=None, experiment=None):

        # super(RegressionTrainer, self).__init__(model, config, experiment_dir)
        # super(RegressionTrainer, self).__init__()
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

        self.experiment = experiment

        self.setup_criterion()
        self.setup_optimizer()
        self.setup_scheduler()

        self.best_accuracy = 0.0
        # self.accuracy_tolerance = getattr(config, 'accuracy_tolerance', 10.0)
        self.accuracy_tolerance = 25

        # self.metric_fn = lambda y_true, y_pred: compute_regression_metrics(
        #     y_true, y_pred, self.accuracy_tolerance
        # )
        self.metric_fn = lambda y_true, y_pred: compute_accuracy(
            y_true, y_pred, self.accuracy_tolerance
        )

    def setup_experiment(self, experiment_name=None, base_dir=None):

        experiment_name = experiment_name or getattr(self.config, 'experiment_name', 'unnamed_experiment')
        base_dir = base_dir or getattr(self.config, 'experiment_base_dir', 'experiments')

        self.experiment = ExperimentManager(experiment_name, base_dir)
        self.experiment.save_config(self.config)

        return self.experiment

    def setup_criterion(self):

        loss_type = getattr(self.config, 'loss_type', 'mape').lower()

        if loss_type == 'mape':
            print("Using MAPE Loss")
            epsilon = getattr(self.config, 'loss_epsilon', 1e-5)
            self.criterion = MapeLoss(epsilon=epsilon)
        elif loss_type == 'mae' or loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'huber':
            delta = getattr(self.config, 'huber_delta', 1.0)
            self.criterion = nn.HuberLoss(delta=delta, reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction='none')

    def setup_optimizer(self):

        optimizer_name = getattr(self.config, 'optimizer', 'adam').lower()

        if optimizer_name == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

    def setup_scheduler(self):

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
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.patience // 2,
                # verbose=True
            )

    def train(self, train_loader, val_loader, num_epochs=None, resume=False, checkpoint_path=None):

        num_epochs = num_epochs or self.config.epochs

        if resume:
            self._resume_checkpoint(checkpoint_path)

        # start_time = time.time()
        print(f"Starting training from epoch {self.start_epoch} to {num_epochs}")

        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch

            train_metrics, train_batch_result = self.train_epoch(train_loader)

            val_metrics = self.validate(val_loader)
            """
                metrics = {
                        "loss": avg_loss,
                        "accuracy": current_accuracy,
                        "is_best_accuracy": True
                    }
            """

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # check if this epoch's validation is the minimum loss
            is_best = val_metrics["loss"] < self.best_metric
            if is_best:
                self.best_metric = val_metrics["loss"]
                self.best_epoch = epoch
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # save checkpoint
            if (epoch + 1) % self.config.save_freq == 0 or is_best:
                self._save_checkpoint(epoch, train_metrics, val_metrics, is_best)

            loss_stats = {
                "instruction_avg_loss": train_batch_result.get_instruction_avg_loss(),
                "instruction_counts": train_batch_result.instruction_counts,
                "block_length_avg_loss": train_batch_result.get_block_length_avg_loss(),
                "block_length_counts": train_batch_result.block_lengths_counts
            }

            if hasattr(self, 'experiment') and self.experiment:
                self.experiment.save_loss_stats(loss_stats, epoch)
                # self.experiment.visualize_epoch_stats(loss_stats, epoch)

            print(f"Epoch {epoch }/{num_epochs} - "
                  f"Train Loss: {train_metrics['loss']:.6f} - "
                  f"Val Loss: {val_metrics['loss']:.6f}" +
                  (f" - Best Val Loss: {self.best_metric:.6f} (Epoch {self.best_epoch})" if is_best else ""))

            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self._plot_progress()

            # early stopping
            if self.early_stopping_counter >= self.config.patience and epoch > 10:
                print(f"Early stopping: Validation loss did not improve for {self.config.patience} epochs")
                break

        # training_time = time.time() - start_time
        # print(f"Training completed! Total time: {training_time:.2f} seconds")
        print(f"Best validation loss: {self.best_metric:.6f} at Epoch {self.best_epoch}")

        self._plot_progress()

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch
        }

    def train_epoch(self, train_loader):

        self.model.train()
        batch_result = BatchResult()
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.epochs}")

        for batch in progress_bar:
            x = batch['X'].to(self.device)
            y = batch['Y'].to(self.device)
            instruction_count = batch.get('instruction_count', None)

            self.optimizer.zero_grad()

            output = self.model(x)
            loss = self.criterion(output, y)  # [batch_size]

            # for i in torch.where(loss > 10)[0]:
            #     print("x:", x["x"][i])
            #     print("y:", y)
            #     print("output:", output)
            #     print("loss:", loss)

            mean_loss = torch.mean(loss)
            mean_loss.backward()

            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            # collect batch statistics
            if self.config.model_type in ['lstm','transformer']:
                for i in range(len(output)):  # batch size
                    instructions = []   # record instruction type of each bb

                    if instruction_count is not None:
                        valid_count = instruction_count[i].item()

                        for j in range(min(valid_count, self.config.max_instr_count)):
                            # instructions = [73, 5, 24, 6, 30, 7, 4]

                            if self.config.model_type == 'transformer':
                                instr_tokens = [t.item() for t in x['x'][i, j] if t.item() != 0]
                            elif self.config.model_type == 'lstm':
                                instr_tokens = [t.item() for t in x[i, j] if t.item() != 0]
                            if instr_tokens:
                                instructions.append(instr_tokens[0])
                    batch_result.add_sample(
                        prediction=output[i].item(),
                        measured=y[i].item(),
                        loss=loss[i].item(),
                        instructions=instructions,
                        block_len=instruction_count[i].item())

            elif self.config.model_type == 'gnn':

                graph_list = batch['X'].to_data_list()
                for i, graph in enumerate(graph_list):

                    batch_result.add_sample(
                        prediction=output[i].item(),
                        measured=y[i].item(),
                        loss=loss[i].item(),
                        instructions=graph.instruction_token_ids.tolist(),
                        block_len=instruction_count[i].item())

            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")

            progress_bar.set_postfix({"loss": mean_loss.item()})

        metrics = batch_result.compute_metrics(self.accuracy_tolerance)

        self.train_losses.append(metrics["loss"])
        '''{
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"]
        }'''

        current_lr = self.optimizer.param_groups[0]['lr']
        metrics["lr"] = current_lr
        self.learning_rates.append(current_lr)

        print(f"\nTraining Statistics - Epoch {self.current_epoch}:")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  Accuracy: {metrics.get(f'accuracy', 0):.6f}")

        if hasattr(self, 'experiment') and self.experiment is not None:
            self.experiment.log_metrics(metrics, self.current_epoch, prefix="train_")

        return metrics, batch_result

    def validate(self, val_loader, epoch=None):

        if epoch is None:
            epoch = self.current_epoch

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        pred = []
        true = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x = batch['X'].to(self.device)
                y = batch['Y'].to(self.device)

                output = self.model(x)
                loss = self.criterion(output, y)

                # for i in torch.where(loss > 10)[0]:
                #     print("x:", x["x"][i])
                #     print("y:", y[i])
                #     print("output:", output[i])

                total_loss += torch.sum(loss).item()
                total_samples += len(x)
                pred.extend(output.tolist())
                true.extend(y.tolist())

        avg_loss = total_loss / total_samples
        current_accuracy = compute_accuracy(true, pred, self.accuracy_tolerance)
        metrics = {
            "loss": avg_loss,
            "accuracy": current_accuracy
        }
        self.val_losses.append(avg_loss)

        print(f"\nValidation Results - Epoch {epoch}:")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  Accuracy: {current_accuracy:.6f}")

        # check if this epoch's validation is the best accuracy
        is_best_accuracy = current_accuracy > self.best_accuracy
        if is_best_accuracy:
            self.best_accuracy = current_accuracy
            metrics["is_best_accuracy"] = True

        self.experiment.log_metrics(metrics, epoch, prefix="val_")

        return metrics

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
