import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils.metrics import compute_regression_metrics, MapeLoss, BatchResult, correct_regression
import time
import os

class RegressionTrainer(BaseTrainer):

    def __init__(self, model, config, experiment_dir=None, experiment=None):

        super(RegressionTrainer, self).__init__(model, config, experiment_dir)
        self.experiment = experiment

        self.setup_criterion()
        self.setup_optimizer()
        self.setup_scheduler()

        self.best_accuracy = 0.0
        # self.accuracy_tolerance = getattr(config, 'accuracy_tolerance', 10.0)
        self.accuracy_tolerance = 25

        self.metric_fn = lambda y_true, y_pred: compute_regression_metrics(
            y_true, y_pred, self.accuracy_tolerance
        )

    def setup_criterion(self):

        loss_type = getattr(self.config, 'loss_type', 'mape').lower()

        if loss_type == 'mape':
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
                verbose=True
            )

    def train_epoch(self, train_loader):

        self.model.train()
        batch_result = BatchResult()
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}")

        """
        batch {'X': tensor([[[ 73,   5,  29,  ...,   7,   4,   0],
         [ 73,   5,  20,  ...,   7,   4,   0],
         [  0,   5,  19,  ...,   4,   0,   0],
         ...,
         [  0,   0,   0,  ...,   0,   0,   0],
         [  0,   0,   0,  ...,   0,   0,   0],
         [  0,   0,   0,  ...,   0,   0,   0]],

        [[185,   5,  24,  ...,   7,   4,   0],
         [209,   5,  27,  ...,   7,   4,   0],
         [217,   5,  27,  ...,   7,   4,   0],
         ...,
         [  0,   0,   0,  ...,   0,   0,   0],
         [  0,   0,   0,  ...,   0,   0,   0],
         [  0,   0,   0,  ...,   0,   0,   0]],

        [[205,   5,  21,  ...,   7,   4,   0],
         [ 73,   5,  24,  ...,   7,   4,   0],
         [205,   5,  24,  ...,   7,   4,   0],
         ...,
         [  0,   0,   0,  ...,   0,   0,   0],
         [  0,   0,   0,  ...,   0,   0,   0],
         [  0,   0,   0,  ...,   0,   0,   0]],

        [[183,   5,  24,  ...,   7,   4,   0],
         [183,   5,  19,  ...,   7,   4,   0],
         [  0,   5,  21,  ...,   4,   0,   0],
         ...,
         [  0,   0,   0,  ...,   0,   0,   0],
         [  0,   0,   0,  ...,   0,   0,   0],
         [  0,   0,   0,  ...,   0,   0,   0]]]), 
         
         'instruction_count': tensor([3, 3, 3, 5]), 
         
         'Y': tensor([3., 3., 2., 5.]), 
         'instruction_text': [b'["addi  s4,s2,1", "addi  a1,zero,37", "mv   a0,s4"]', 
         b'["lhu    a5,102(s1)", "slliw  s2,a0,16", "sraiw  s2,s2,16"]', 
         b'["sd   a2,0(a3)", "addi    a5,s1,728", "sd   a5,936(s1)"]', 
         b'["ld   a5,16(s0)", "ld    a0,0(s9)", "mv   a2,zero", "auipc  a1,21", "addi  a1,a1,576"]']}

        """

        for batch in progress_bar:
            x = batch['X'].to(self.device)
            y = batch['Y'].to(self.device)
            instruction_count = batch.get('instruction_count', None)

            self.optimizer.zero_grad()

            output = self.model(x)
            loss = self.criterion(output, y)  # [batch_size]
            mean_loss = torch.mean(loss)
            mean_loss.backward()

            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            # collect batch statistics
            for i in range(len(output)):  # batch size

                instructions = []   # record instruction type of each bb
                block_len = None

                if instruction_count is not None:
                    valid_count = instruction_count[i].item()
                    block_len = valid_count

                    for j in range(valid_count):
                        # instruction = [73, 5, 24, 6, 30, 7, 4]
                        instr_tokens = [t.item() for t in x[i, j] if t.item() != 0]
                        if instr_tokens:
                            instructions.append(instr_tokens[0])

                batch_result.add_sample(
                    prediction=output[i].item(),
                    measured=y[i].item(),
                    loss=loss[i].item(),
                    instructions=instructions,
                    block_len=block_len
                )

            progress_bar.set_postfix({"loss": mean_loss.item()})

            self.global_step += 1

        metrics = batch_result.compute_metrics(self.accuracy_tolerance)
        self.train_losses.append(metrics["loss"])
        '''{
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"]
        }'''

        current_lr = self.optimizer.param_groups[0]['lr']
        metrics["lr"] = current_lr
        self.learning_rates.append(current_lr)

        print(f"\nTraining Statistics - Epoch {self.current_epoch + 1}:")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  Accuracy: {metrics.get(f'accuracy', 0):.6f}")

        if hasattr(self, 'experiment') and self.experiment is not None:
            self.experiment.log_metrics(metrics, self.current_epoch, prefix="train_")

        return metrics, batch_result

    def train(self, train_loader, val_loader, num_epochs=None, resume=False, checkpoint_path=None):

        num_epochs = num_epochs or self.config.epochs

        if resume:
            self._resume_checkpoint(checkpoint_path)

        start_time = time.time()
        print(f"Starting training---------------------------\nFrom epoch {self.start_epoch + 1} to {num_epochs}")

        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch

            train_metrics, train_batch_result = self.train_epoch(train_loader)

            val_metrics = self.validate(val_loader)

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

            instruction_stats = {
                    "instruction_avg_loss": train_batch_result.get_instruction_avg_loss(),
                    "instruction_counts": train_batch_result.instruction_counts
            }

            block_length_stats = {
                "block_length_avg_loss": train_batch_result.get_block_length_avg_loss(),
                "block_length_counts": train_batch_result.block_lengths_counts
            }

            if hasattr(self, 'experiment') and self.experiment:
                self.experiment.save_instruction_stats(instruction_stats, epoch)
                self.experiment.save_block_length_stats(block_length_stats, epoch)

                self.experiment.visualize_epoch_stats(
                    instruction_stats,
                    block_length_stats,
                    epoch
                )

            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_metrics['loss']:.6f} - "
                  f"Val Loss: {val_metrics['loss']:.6f}" +
                  (f" - Best Val Loss: {self.best_metric:.6f} (Epoch {self.best_epoch + 1})" if is_best else ""))

            # 绘制训练进度
            # if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            #     self._plot_progress()

            # early stopping
            if self.early_stopping_counter >= self.config.patience:
                print(f"Early stopping: Validation loss did not improve for {self.config.patience} epochs")
                break

        training_time = time.time() - start_time
        print(f"Training completed! Total time: {training_time:.2f} seconds")
        print(f"Best validation loss: {self.best_metric:.6f} at Epoch {self.best_epoch + 1}")

        # 绘制最终训练进度
        # self._plot_progress()

        # # 加载最佳模型
        # best_checkpoint_path = os.path.join(self.checkpoint_dir, "model_best.pth")
        # if os.path.exists(best_checkpoint_path):
        #     self._resume_checkpoint(best_checkpoint_path, only_model=True)
        #     print(f"已加载最佳模型 (Epoch {self.best_epoch + 1})")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch
        }

    def validate(self, val_loader, epoch=None):

        if epoch is None:
            epoch = self.current_epoch

        self.model.eval()
        batch_result = BatchResult()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x = batch['X'].to(self.device)
                y = batch['Y'].to(self.device)
                instruction_count = batch.get('instruction_count', None)

                output = self.model(x)
                loss = self.criterion(output, y)

                for i in range(len(output)):

                    instructions = []
                    block_len = None

                    if instruction_count is not None:
                        valid_count = instruction_count[i].item()
                        block_len = valid_count

                        for j in range(valid_count):
                            instr_tokens = [t.item() for t in x[i, j] if t.item() != 0]
                            if instr_tokens:
                                instructions.append(instr_tokens[0])

                    batch_result.add_sample(
                        prediction=output[i].item(),
                        measured=y[i].item(),
                        loss=loss[i].item(),
                        instructions=instructions,
                        block_len=block_len
                    )

        metrics = batch_result.compute_metrics(self.accuracy_tolerance)
        self.val_losses.append(metrics["loss"])
        current_accuracy = metrics.get("accuracy", 0)

        print(f"\nValidation Results - Epoch {epoch + 1}:")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  Accuracy: {current_accuracy:.6f}")

        # check if this epoch's validation is the best accuracy
        is_best_accuracy = current_accuracy > self.best_accuracy
        if is_best_accuracy:
            self.best_accuracy = current_accuracy
            metrics["is_best_accuracy"] = True

        self.experiment.log_metrics(metrics, epoch, prefix="val_")

        return metrics