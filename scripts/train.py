# !/usr/bin/env python
"""
模型训练脚本 - 训练RISC-V吞吐量预测模型
"""
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from config import get_config
from data import get_dataloader
from models import get_model
from trainers import Trainer
from utils import set_seed, ExperimentManager


def main():
    parser = argparse.ArgumentParser(description="RISC-V Instruction Throughput Prediction Model Training")

    # Data arguments
    parser.add_argument("--train_data", type=str, default="data/train_data.json", help="Path to training data")
    parser.add_argument("--val_data", type=str, default="data/val_data.json", help="Path to validation data")

    # Model arguments
    parser.add_argument("--model_type", type=str, default="transformer",
                        choices=["transformer", "lstm", "gnn"], help="Model type")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    # parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")
    # parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads (transformer only)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine_warmup", choices=["plateau", "cosine", "cosine_warmup", "step"],
                        help="Learning rate scheduler")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="default", help="Experiment name")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Training device")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading threads")

    args = parser.parse_args()
    experiment_manager = ExperimentManager(args.experiment_name, args.output_dir)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", args.device)

    config = get_config(
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        # num_layers=args.num_layers,
        # num_heads=args.num_heads,
        # dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        clip_grad_norm=args.clip_grad_norm,
        device=args.device,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=experiment_manager.experiment_dir,
        experiment_name=args.experiment_name,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        seed=args.seed,
    )
    print("config", config.num_layers)

    experiment_manager.save_config(config)

    train_loader = get_dataloader(
        args.model_type,
        args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory = True,  # 使用pinned memory加速GPU传输
        prefetch_factor = 2,  # 每个worker预加载的批次数
        persistent_workers = True
    )

    val_loader = get_dataloader(
        args.model_type,
        args.val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,  # 使用pinned memory加速GPU传输
        prefetch_factor=2,  # 每个worker预加载的批次数
        persistent_workers=True
    )

    model = get_model(config)
    print(f"Create model: {args.model_type.upper()}, Number of parameters: {model.count_parameters():,}")

    trainer = Trainer(model, config, experiment_manager.experiment_dir, experiment_manager)
    # trainer.setup_experiment(args.experiment_name, args.output_dir)

    # print(f"Starting training..., device: {device}")
    experiment_manager.start(args.train_data, args.val_data, train_loader.dataset, val_loader.dataset)
    history = trainer.train(train_loader, val_loader)
    # return {
    #     "train_losses": self.train_losses,
    #     "val_losses": self.val_losses,
    #     "learning_rates": self.learning_rates,
    #     "best_metric": self.best_metric,
    #     "best_epoch": self.best_epoch
    # }

    experiment_manager.history = history
    experiment_manager.save_history()

    # print(f"Training completed! Best validation loss: {history['best_metric']:.6f} at Epoch {history['best_epoch'] + 1}")
    experiment_manager.save_summary({
        'model_type': args.model_type,
        'best_val_loss': history['best_metric'],
        'best_epoch': history['best_epoch'] + 1,
        'parameters': model.count_parameters(),
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset)
    })

    experiment_manager.finish()
    print(f"Finished! Results saved to: {experiment_manager.experiment_dir}")


if __name__ == "__main__":
    main()