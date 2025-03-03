from typing import Dict, Type, Optional

from .base_model import BaseModel
from .transformer_model import TransformerModel
from .lstm_model import LSTMModel
from .gnn_model import GNNModel


MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "transformer": TransformerModel,
    "lstm": LSTMModel,
    "gnn": GNNModel,
    "ithemal": LSTMModel  # 使用同一个LSTMModel类但配置不同
}
def register_model(name: str, model_class: Type[BaseModel]) -> None:
    """
    注册新模型到注册表
    
    Args:
        name: 模型名称
        model_class: 模型类
    """
    global MODEL_REGISTRY
    if name in MODEL_REGISTRY:
        raise ValueError(f"模型 {name} 已经注册")
    
    MODEL_REGISTRY[name] = model_class
    print(f"模型 {name} 已注册")

def get_model(config, model_type: Optional[str] = None) -> BaseModel:
    """
    根据配置创建模型实例
    
    Args:
        config: 配置对象
        model_type: 模型类型，如果为None则使用config中的model_type
        
    Returns:
        模型实例
    """
    model_type = model_type or config.model_type
    
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(config)
    
    return model

def list_available_models() -> Dict[str, Type[BaseModel]]:
    """
    列出所有可用的模型
    
    Returns:
        模型注册表
    """
    return MODEL_REGISTRY.copy()
