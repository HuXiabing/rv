from .tokenizer import RISCVTokenizer
from .dataset import RISCVDataset, get_dataloader

## 创建配置
#from config import get_config
#config = get_config(model_type="transformer")
#
## 数据处理
#from data import RISCVDataProcessor
#processor = RISCVDataProcessor(config)
#
## 处理原始数据
#processor.process_raw_data()
#
## 分别处理训练集和验证集
#processor.process_data_to_h5(output_path="data/train_data.h5", is_training=True)
#processor.process_data_to_h5(output_path="data/val_data.h5", is_training=False)
#
## 创建数据加载器
#from data import get_dataloader
#train_loader = get_dataloader("data/train_data.h5", batch_size=config.batch_size)
#val_loader = get_dataloader("data/val_data.h5", batch_size=config.batch_size, shuffle=False)

