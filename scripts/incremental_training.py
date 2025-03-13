#!/usr/bin/env python
"""
Incremental Training Script - Incremental Training Based on an Existing Model
"""
import os
import sys
import argparse
import glob
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from config import get_config
from data import get_dataloader
from models import get_model
from trainers import RegressionTrainer
from utils import set_seed, ExperimentManager


def find_latest_model(experiment_dir):

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    best_model_path = os.path.join(checkpoint_dir, "model_best.pth")
    if os.path.exists(best_model_path):
        return best_model_path

    latest_model_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    if os.path.exists(latest_model_path):
        return latest_model_path

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if checkpoint_files:
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        return checkpoint_files[0]

    return None

def main():

    parser = argparse.ArgumentParser(description="Incremental Training Based on an Existing Model")

    # Model Parameters
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the checkpoint of the trained model. Defaults to the most recently trained model.")
    parser.add_argument("--original", type=str, default=None,
                        help="Original experiment directory, used to automatically find the latest model (if model_path is not specified).")

    # Data Parameters
    parser.add_argument("--train_data", type=str, default="data/train_data.h5", help="Path to the new training data (HDF5).")
    parser.add_argument("--val_data", type=str, default="data/val_data.h5", help="Path to the validation data (HDF5).")

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for incremental training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    # parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay.")
    # parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    # parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping threshold.")
    parser.add_argument("--restart_optimizer", action="store_true", help="Whether to reinitialize the optimizer.")

    # Output Parameters
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory.")
    parser.add_argument("--experiment_name", type=str, default="incremental", help="Experiment name.")


    # Other Parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading threads.")

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    if args.model_path is None:
        if args.original is None:
            # if both model_path and original dir are not specified, try to find the latest model in the experiments directory
            if os.path.exists("experiments"):
                experiments = [d for d in os.listdir("experiments")
                               if os.path.isdir(os.path.join("experiments", d))]

                if experiments:
                    experiments.sort(key=lambda d: os.path.getmtime(os.path.join("experiments", d)),
                                     reverse=True)
                    args.original = os.path.join("experiments", experiments[0])
                    print(f"Automatically using the latest experiment: {args.original}")
                else:
                    parser.error("None of the experiments found in the experiments directory. Please specify --model_path or --original")

        if args.original:
            model_path = find_latest_model(args.original)
            if model_path:
                args.model_path = model_path
                print(f"Using the latest model checkpoint found in {args.original}: {args.model_path}")
            else:
                parser.error(f"No model checkpoint found in {args.original}")

    checkpoint = torch.load(args.model_path, map_location='cpu')

    # Load configuration
    config_dict = checkpoint.get('config', None)
    if config_dict is None:
        raise ValueError(f"There is no configuration information in {args.model_path}")

    config = get_config(**config_dict)
    args.experiment_name = f"{args.experiment_name}_{config.model_type}"

    experiment_manager = ExperimentManager(args.experiment_name, args.output_dir)
    experiment_manager.save_config(config)

    model = get_model(config)

    model_state_dict = model.state_dict()
    for name, param in checkpoint['model_state'].items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"Skipping parameter {name}: Shape mismatch ({param.shape} vs {model_state_dict[name].shape})")
        else:
            print(f"Skipping parameter {name}: Not found in the current model")

    model.load_state_dict(model_state_dict)
    model.to(device)

    print(f"Create model: {config.model_type.upper()}, Number of parameters: {model.count_parameters():,}")

    train_loader = get_dataloader(
        config.model_type,
        args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = get_dataloader(
        config.model_type,
        args.val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    trainer = RegressionTrainer(model, config, experiment_manager.experiment_dir, experiment_manager)

    if not args.restart_optimizer and 'optimizer_state' in checkpoint:
        try:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded the optimizer state from the original model")
        except Exception as e:
            print(f"Failed to load the optimizer state: {e}")
            print("Using a newly initialized optimizer")
    else:
        print("Using a newly initialized optimizer")

    # Start Incremental Training
    # print(f"Training data: {args.train_data}, Number of samples: {len(train_loader.dataset)}")
    # print(f"Validation data: {args.val_data}, Number of samples: {len(val_loader.dataset)}")
    experiment_manager.start(args.train_data, args.val_data, train_loader.dataset, val_loader.dataset)

    history = trainer.train(train_loader, val_loader)

    experiment_manager.history = history
    experiment_manager.save_history()

    experiment_manager.save_summary({
        'model_type': config.model_type,
        'best_val_loss': history['best_metric'],
        'best_epoch': history['best_epoch'] + 1,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'incremental_learning': True
    })

    experiment_manager.finish()
    print(f"Incremental training completed! Model saved at: {experiment_manager.experiment_dir}")
    print(f"Best validation loss: {history['best_metric']:.6f} at Epoch {history['best_epoch'] + 1}")

if __name__ == "__main__":
    main()