import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import json

from .base_model import BaseModel
from .model import RISCVGraniteModel
from .graph_encoder import RISCVGraphEncoder

class GNNModel(BaseModel):

    def __init__(self, config):

        super(GNNModel, self).__init__(config)

        # 创建图编码器
        self.graph_encoder = RISCVGraphEncoder()

        # 创建RISC-V GRANITE模型
        self.model = RISCVGraniteModel(
            node_embedding_dim=config.embed_dim,  # 节点嵌入维度
            edge_embedding_dim=config.embed_dim,  # 边嵌入维度
            global_embedding_dim=config.embed_dim,  # 全局嵌入维度
            hidden_dim=config.hidden_dim,  # 隐藏层维度
            num_message_passing_steps=config.num_layers,  # 消息传递步数
            dropout=config.dropout,  # Dropout
            use_layer_norm=getattr(config, 'use_layer_norm', True),  # 是否使用层归一化
            num_tasks=1,  # 单任务设置
            use_multi_task_decoder=False  # 不使用多任务解码器
        )

    def _convert_to_basic_block(self, x, instruction_count) -> List[List[str]]:
        """
        将模型输入转换为RISC-V基本块格式

        Args:
            x: 输入张量 [batch_size, max_instr_count, max_instr_length]
            instruction_count: 指令数量 [batch_size]

        Returns:
            RISC-V基本块列表
        """
        batch_size = x.size(0)
        basic_blocks = []

        for i in range(batch_size):
            # 确定有效的指令数量
            valid_count = instruction_count[i].item() if instruction_count is not None else x.size(1)

            # 提取并转换指令
            instructions = []
            for j in range(valid_count):
                # 提取非零令牌
                tokens = [t.item() for t in x[i, j] if t.item() != 0]
                if tokens:
                    # 这里我们需要从令牌ID转换回实际的指令文本
                    # 简单起见，我们暂时创建一个占位字符串
                    instr_str = f"instr_{j}"
                    instructions.append(instr_str)

            basic_blocks.append(instructions)

        return basic_blocks


    def forward(self, x, instruction_count=None):
        """

        Args:
            x: 输入数据或批次对象
            instruction_count: 每个样本的指令数量 [batch_size]

        Returns:
            预测的吞吐量值 [batch_size]
        """
        # 检查x是否为字典类型（从dataloader加载的批次）
        if isinstance(x, dict):
            batch_size = x['X'].size(0)
            device = x['X'].device

            # 如果有instruction_text字段，使用它
            if 'instruction_text' in x:
                instruction_texts = x['instruction_text']
                if instruction_count is None and 'instruction_count' in x:
                    instruction_count = x['instruction_count']
            else:
                # 使用X和instruction_count
                instruction_texts = None
                x_tensor = x['X']
                if instruction_count is None and 'instruction_count' in x:
                    instruction_count = x['instruction_count']
        else:
            # 假设x是张量
            batch_size = x.size(0)
            device = x.device
            instruction_texts = None
            x_tensor = x

        results = []

        # 处理每个样本
        for i in range(batch_size):
            # 确定有效的指令数量
            valid_count = instruction_count[i].item() if instruction_count is not None else x_tensor.size(1)

            # 获取指令文本
            if instruction_texts is not None:
                # 解析JSON字符串获取原始指令
                instructions = json.loads(instruction_texts[i])[:valid_count]
            else:
                # 从令牌ID列表构建指令字符串
                instructions = []
                for j in range(valid_count):
                    tokens = [t.item() for t in x_tensor[i, j] if t.item() != 0]
                    if tokens:
                        # 简单拼接令牌ID作为指令文本
                        instr_str = " ".join([str(t) for t in tokens])
                        instructions.append(instr_str)

            if not instructions:
                # 如果没有有效指令，预测为0
                results.append(torch.tensor(0.0, device=device))
                continue

            # 构建图
            graph = self.graph_encoder.build_graph(instructions)

            # 将图移动到同一设备
            graph = graph.to(device)

            # 使用模型预测
            throughput = self.model(graph)

            results.append(throughput)

        # 堆叠所有结果
        return torch.stack(results)