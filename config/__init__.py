from .config import Config
from .defaults import (
    DEFAULT_CONFIG, 
    TRANSFORMER_CONFIG, 
    LSTM_CONFIG, 
    GNN_CONFIG, 
    DATA_CONFIG,
    TRAINING_CONFIG,
    ITHEMAL_CONFIG
)


def get_config(model_type="transformer", **kwargs):
    """获取指定模型类型的配置，并用kwargs更新"""
    config_dict = DEFAULT_CONFIG.copy()

    # 根据模型类型添加特定配置
    if model_type == "transformer":
        config_dict.update(TRANSFORMER_CONFIG)
    elif model_type == "lstm":
        config_dict.update(LSTM_CONFIG)
    elif model_type == "gnn":
        config_dict.update(GNN_CONFIG)
    elif model_type == "ithemal":
        config_dict.update(ITHEMAL_CONFIG)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 使用kwargs更新配置
    config_dict.update(kwargs)

    # 确保model_type正确设置
    config_dict["model_type"] = model_type

    return Config(**config_dict)

