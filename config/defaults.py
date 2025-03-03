
TRANSFORMER_CONFIG = {
    "model_type": "transformer",
    "embed_dim": 512,
    "hidden_dim": 2048,
    "num_layers": 4,
    "num_heads": 8,
    "dropout": 0.1,
    "use_layernorm": True,
    "use_checkpoint": False,
    "use_bb_attn": True,
    "use_seq_attn": True,
    "use_op_attn": True,
    "use_pos_2d": False,
    "handle_neg": False
}

# LSTM模型默认配置 (为BatchRNN更新)
LSTM_CONFIG = {
    "model_type": "lstm",
    "embed_dim": 512,
    "hidden_dim": 512,
    "num_layers": 1,
    "dropout": 0.1,
}

# 添加Ithemal配置
ITHEMAL_CONFIG = {
    "model_type": "ithemal",
    "embed_dim": 512,
    "hidden_dim": 512,
    "num_layers": 1,
    "dropout": 0.1,
}

# # GNN模型默认配置
# GNN_CONFIG = {
#     "model_type": "gnn",
#     "embed_dim": 128,
#     "hidden_dim": 256,
#     "num_layers": 3,
#     "dropout": 0.1,
# }
# GNN模型默认配置 (为GRANITE更新)
GNN_CONFIG = {
    "model_type": "gnn",
    "embed_dim": 256,
    "hidden_dim": 256,
    "num_layers": 8,  # 消息传递步数
    "dropout": 0.1,
    "use_layer_norm": True,
    "lr": 5e-4,  # 针对GNN的学习率调整
}

# 数据处理默认配置
DATA_CONFIG = {
    "max_instr_length": 20,
    "max_instr_count": 20,
    "vocab_size": 2000,
}

# 训练参数默认配置
TRAINING_CONFIG = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "epochs": 50,
    "patience": 5,
    "clip_grad_norm": 1.0,
    "loss_type": "mape",  # 'mse', 'mae', 'huber', 'mape'
    "loss_epsilon": 1e-5,  # 用于MAPE损失函数
    "huber_delta": 1.0,    # 用于Huber损失函数
}
# 合并默认配置
DEFAULT_CONFIG = {
    **DATA_CONFIG,
    **TRAINING_CONFIG,
    "output_dir": "outputs",
    "experiment_name": "default",
    "seed": 42,
    "verbose": True,
    "save_best_only": True,
    "save_freq": 1,
    "max_checkpoints": 3,
}
