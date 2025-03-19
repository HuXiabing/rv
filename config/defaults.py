TRANSFORMER_CONFIG = {
    "model_type": "transformer",
    "embed_dim": 512,
    "hidden_dim": 2048,
    "num_layers": 4,
    "num_heads": 8,
    "dropout": 0.1,
    "use_layernorm": False,
    "use_checkpoint": False,
    "use_bb_attn": False,
    "use_seq_attn": True,
    "use_op_attn": False,
    "use_pos_2d": False,
    "handle_neg": False
}

ITHEMAL_CONFIG = {
    "model_type": "ithemal",
    "embed_dim": 512,
    "hidden_dim": 512,
    "num_layers": 1,
    "dropout": 0.1,
}

GNN_CONFIG = {
    "model_type": "gnn",
    "embed_dim": 256,
    "hidden_dim": 256,
    "num_layers": 8,  # message passing layers
    "dropout": 0.1,
    "use_layer_norm": True,
    "lr": 2e-5,
}

# default data configuration
DATA_CONFIG = {
    "max_instr_length": 8,
    "max_instr_count": 64,
    "vocab_size": 256,
}

# default training configuration
TRAINING_CONFIG = {
    "lr": 2e-5,
    "weight_decay": 1e-5,
    "batch_size": 8,
    "epochs": 50,
    "patience": 3,
    "clip_grad_norm": 1.0,
    "loss_type": "mape",  # 'mse', 'mae', 'huber', 'mape'
    "loss_epsilon": 1e-5,  # for MAPE
    "huber_delta": 1.0,    # for Huber
}

# merge all configurations
DEFAULT_CONFIG = {
    **DATA_CONFIG,
    **TRAINING_CONFIG,
    "output_dir": "experiments",
    "experiment_name": "test",
    "seed": 42,
    "verbose": True,
    "save_best_only": True,
    "save_freq": 1,
    "max_checkpoints": 3,
}
