Quick Start
1. 数据预处理
将原始RISC-V指令数据转换为模型可用的格式：
python main.py preprocess --raw_data data/labeled_data.json --output_dir data
预处理将生成训练集、验证集和测试集的HDF5文件。
2. 模型训练
训练一个Transformer模型：
python main.py train --model_type transformer --train_data data/train_data.h5 --val_data data/val_data.h5 --experiment_name transformer_v1
训练过程中的检查点、日志和可视化结果将保存在experiments/{实验名称}目录中。
3. 模型评估
在测试集上评估模型：
python main.py evaluate --model_path experiments/transformer_v1/checkpoints/model_best.pth --test_data data/test_data.h5 --output_dir evaluation/transformer_v1
4. 模型推理
使用训练好的模型进行推理：
python main.py predict --model_path experiments/transformer_v1/checkpoints/model_best.pth --input_json data/new_samples.json --output predictions.json
5. 恢复训练
从检查点恢复训练：
python main.py resume --checkpoint experiments/transformer_v1/checkpoints/checkpoint_epoch_10.pth --additional_epochs 20
模型架构
Transformer模型
基于自注意力机制的模型，适合捕捉指令间的长距离依赖关系。
LSTM模型
使用层次化双向LSTM处理指令序列，适合序列建模。
GNN模型
将指令序列建模为图结构，使用图注意力网络处理，适合捕捉指令间的复杂交互。
自定义配置
您可以通过命令行参数自定义模型和训练参数：
python main.py train \
  --model_type transformer \
  --embed_dim 256 \
  --hidden_dim 512 \
  --num_layers 6 \
  --num_heads 8 \
  --dropout 0.1 \
  --batch_size 64 \
  --epochs 100 \
  --lr 0.0001
数据格式
