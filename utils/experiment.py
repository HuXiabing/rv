import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import torch
import collections

class ExperimentManager:

    def __init__(self, experiment_name: str, base_dir: str = "experiments"):

        self.experiment_name = experiment_name
        self.base_dir = base_dir

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"

        self.experiment_dir = os.path.join(base_dir, self.experiment_id)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.setup_directories()
        self.setup_logger()
        
        # metrics
        self.metrics = {}
        # self.history = {
        #     "train_losses": [],
        #     "val_losses": [],
        #     "learning_rates": [],
        #     "metrics": {}
        # }
        self.history = {}
        # return {
        #     "train_losses": self.train_losses,
        #     "val_losses": self.val_losses,
        #     "learning_rates": self.learning_rates,
        #     "best_metric": self.best_metric,
        #     "best_epoch": self.best_epoch
        # }

        self.start_time = time.time()
        self.logger.info(f"Experiment created: {self.experiment_id}")
    
    def setup_directories(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def setup_logger(self):

        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:

            log_file = os.path.join(self.log_dir, "experiment.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def save_config(self, config):

        config_path = os.path.join(self.experiment_dir, "config.json")
        
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = dict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        self.logger.info(f"Configuration saved to {config_path}")

    def load_config(self, config_class=None):

        config_path = os.path.join(self.experiment_dir, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        if config_class is not None:
            config = config_class(**config_dict)
        else:
            config = config_dict

        self.logger.info(f"Configuration loaded from {config_path}")
        return config

    def log_metrics(self, metrics: Dict[str, Any], epoch: int, prefix: str = ""):
        """
        Log training/validation metrics

        Args:
            metrics: Dictionary of metrics
            {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            ...
            }
            epoch: Current epoch (e.g., epoch)
            prefix: Metric prefix (e.g., 'train_' or 'val_')
        """
        metrics_str_parts = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                # If the value is numeric, use the .6f format
                metrics_str_parts.append(f"{name}: {value:.6f}")
            else:
                metrics_str_parts.append(f"{name}: {value}")

        metrics_str = ", ".join(metrics_str_parts)

        self.logger.info(f"Epoch {epoch} - {prefix}metrics: {metrics_str}")
        # self.save_metrics()

    def save_history(self):

        history_path = os.path.join(self.log_dir, "history.json")

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def save_summary(self, summary_data: Dict[str, Any]):

        os.makedirs(self.experiment_dir, exist_ok=True)
        summary_path = os.path.join(self.experiment_dir, "summary.json")

        try:
            summary_data['duration'] = time.time() - self.start_time

            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=4)

            self.logger.info(f"Experiment summary saved to {summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving experiment summary: {e}")
            raise
    
    def finish(self):
        duration = time.time() - self.start_time
        self.logger.info(f"Experiment completed. Best validation loss: {self.history['best_metric']:.6f} at Epoch "
              f"{self.history['best_epoch']}. Total time: {duration:.2f} seconds")

    def start(self,train_data, val_data, train_dataset, val_dataset):
        self.logger.info(f"Training data: {train_data}, Number of samples: {len(train_dataset)}")
        self.logger.info(f"Validation data: {val_data}, Number of samples: {len(val_dataset)}")

    def save_instruction_stats(self, instruction_stats, epoch):
        """
        Save instruction type statistics

        Args:
            instruction_stats: Dictionary of instruction statistics
            epoch: Current epoch
        """
        stats_dir = os.path.join(self.experiment_dir, "statistics")
        os.makedirs(stats_dir, exist_ok=True)

        stats_path = os.path.join(stats_dir, f"instruction_statistics_epoch_{epoch + 1}.json")

        with open(stats_path, 'w') as f:
            json.dump(instruction_stats, f, indent=4)

        self.logger.info(f"Instruction statistics saved to {stats_path}")

    def save_block_length_stats(self, block_length_stats, epoch):
        """
        Save basic block length statistics

        Args:
            block_length_stats: Dictionary of basic block length statistics
            epoch: Current epoch
        """
        stats_dir = os.path.join(self.experiment_dir, "statistics")
        os.makedirs(stats_dir, exist_ok=True)

        stats_path = os.path.join(stats_dir, f"block_length_statistics_epoch_{epoch + 1}.json")

        with open(stats_path, 'w') as f:
            json.dump(block_length_stats, f, indent=4)

        self.logger.info(f"Basic block length statistics saved to {stats_path}")

    def save_loss_stats(self, loss_stats, epoch):
        # loss_stats = {
        #     "instruction_avg_loss": train_batch_result.get_instruction_avg_loss(),
        #     "instruction_counts": train_batch_result.instruction_counts,
        #     "block_length_avg_loss": train_batch_result.get_block_length_avg_loss(),
        #     "block_length_counts": train_batch_result.block_lengths_counts
        # }
        token = torch.load("data/vocab.dump")
        loss_stats["type_avg_loss"] = {key: loss_stats["instruction_avg_loss"][value] for key, value in token.items()
                                       if value in loss_stats["instruction_avg_loss"]}
        loss_stats["type_counts"] = {key: loss_stats["instruction_counts"][value] for key, value in token.items()
                                     if value in loss_stats["instruction_counts"]}

        instr_categories = {
            # Arithmetic
            'add': 'arithmetic', 'addi': 'arithmetic',
            'addw': 'arithmetic', 'addiw': 'arithmetic',
            'sub': 'arithmetic', 'subw': 'arithmetic',
            'lui': 'arithmetic', 'auipc': 'arithmetic', # 8

            # Shifts
            'sll': 'shifts', 'sllw': 'shifts',
            'slli': 'shifts', 'slliw': 'shifts',
            'srl': 'shifts', 'srlw': 'shifts',
            'srli': 'shifts', 'srliw': 'shifts',
            'sra': 'shifts', 'sraw': 'shifts',
            'srai': 'shifts', 'sraiw': 'shifts',  # 12

            # Logical
            'or': 'logical', 'ori': 'logical',
            'xor': 'logical', 'xori': 'logical',
            'and': 'logical', 'andi': 'logical', # 6

            # Comparison instructions
            'slt': 'compare', 'slti': 'compare', 'sltu': 'compare', 'sltiu': 'compare',  # 4

            # Multiplication
            'mul': 'mul', 'mulh': 'mul', 'mulhu': 'mul', 'mulhsu': 'mul', 'mulw': 'mul', # 5

            # Division
            'div': 'div', 'divu': 'div', 'divw': 'div', 'divuw': 'div', # 4

            # Remainder
            'rem': 'rem', 'remu': 'rem', 'remw': 'rem', 'remuw': 'rem', # 4

            # Load instructions
            'lb': 'load', 'lh': 'load', 'lw': 'load', 'ld': 'load',
            'lbu': 'load', 'lhu': 'load', 'lwu': 'load',  # 7

            # Store instructions
            'sb': 'store', 'sh': 'store', 'sw': 'store', 'sd': 'store' # 4
        }
        type_order = ['arithmetic', 'shifts', 'logical', 'compare', 'mul', 'div', 'rem', 'load', 'store']
        # Initialize category loss and count accumulators
        loss_stats["category_loss_sum"] = {cat: 0.0 for cat in type_order}
        loss_stats["category_count_sum"] = {cat: 0 for cat in type_order}

        # Aggregate loss and counts by category
        for instr, loss in loss_stats['type_avg_loss'].items():
            category = instr_categories.get(instr)
            if category and category in type_order:
                count = loss_stats['type_counts'].get(instr, 1)
                loss_stats["category_loss_sum"][category] += loss
                loss_stats["category_count_sum"][category] += count

        loss_stats["block_length_avg_loss_sorted"] = collections.OrderedDict(
            sorted({int(k): v for k, v in loss_stats["block_length_avg_loss"].items()}.items())
        )

        loss_stats["block_length_counts_sorted"] = collections.OrderedDict(
            sorted({int(k): v for k, v in loss_stats["block_length_counts"].items()}.items())
        )

        stats_dir = os.path.join(self.experiment_dir, "statistics")
        os.makedirs(stats_dir, exist_ok=True)

        stats_path = os.path.join(stats_dir, f"{loss_stats['prefix']}_loss_stats_epoch_{epoch}.json")

        with open(stats_path, 'w') as f:
            json.dump(loss_stats, f, indent=4)

        self.logger.info(f"Loss statistics saved to {stats_path}")

    def visualize_epoch_stats(self, loss_stats, epoch):
        """
        Generate statistical visualizations for the current epoch

        Args:
            instruction_stats: Dictionary of instruction statistics
            block_length_stats: Dictionary of basic block length statistics
            epoch: Current epoch

        instruction_stats = {
                    "instruction_avg_loss": train_batch_result.get_instruction_avg_loss(),
                    "instruction_counts": train_batch_result.instruction_counts
            }
        block_length_stats = {
                "block_length_avg_loss": train_batch_result.get_block_length_avg_loss(),
                "block_length_counts": train_batch_result.block_lengths_counts
            }
        """
        from utils.visualize import plot_instruction_losses, plot_block_length_losses
        from utils.analysis import analyze_instruction_statistics, analyze_block_length_statistics

        viz_dir = os.path.join(self.experiment_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # analyze statistics for each epoch
        analysis_output_dir = os.path.join(self.experiment_dir, f"analysis_epoch_{epoch + 1}")

        instruction_vec = analyze_instruction_statistics(
            loss_stats["instruction_avg_loss"],
            mapping_dict_path="data/vocab.dump",
            output_dir=analysis_output_dir
        )

        block_dict = analyze_block_length_statistics(
            loss_stats["block_length_avg_loss"],
            output_dir=analysis_output_dir
        )

        # save analysis summary
        analysis_summary = {
            "instruction_vec": instruction_vec,
            "block_dict": {str(k): v for k, v in block_dict.items()}
        }

        with open(os.path.join(analysis_output_dir, "analysis_summary.json"), 'w') as f:
            json.dump(analysis_summary, f, indent=2)

        # plot instruction type loss distribution
        instr_viz_path = os.path.join(viz_dir, f"instruction_losses_epoch_{epoch + 1}.png")
        plot_instruction_losses(
            loss_stats["instruction_avg_loss"],
            loss_stats["instruction_counts"],
            save_path=instr_viz_path,
            title=f"Average Loss by Instruction Type (Epoch {epoch + 1})"
        )

        # plot basic block length loss distribution
        block_viz_path = os.path.join(viz_dir, f"block_length_losses_epoch_{epoch + 1}.png")
        plot_block_length_losses(
            loss_stats["block_length_avg_loss"],
            loss_stats["block_length_counts"],
            save_path=block_viz_path,
            title=f"Average Loss by Basic Block Length (Epoch {epoch + 1})"
        )

        self.logger.info(f"Statistical visualizations and analysis generated for epoch {epoch + 1}")


