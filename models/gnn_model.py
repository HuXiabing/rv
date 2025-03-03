# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, Optional, List
#
# from .base_model import BaseModel
# from .layers import GraphAttentionLayer
#
# class GNNModel(BaseModel):
#     """基于图神经网络的RISC-V指令集吞吐量预测模型"""
#
#     def __init__(self, config):
#         """
#         初始化GNN模型
#
#         Args:
#             config: 配置对象
#         """
#         super(GNNModel, self).__init__(config)
#
#         # 嵌入层
#         self.token_embedding = nn.Embedding(
#             config.vocab_size,
#             config.embed_dim,
#             padding_idx=0
#         )
#
#         # 图神经网络层
#         self.gnn_layers = nn.ModuleList([
#             GraphAttentionLayer(
#                 in_features=config.embed_dim if i == 0 else config.hidden_dim,
#                 out_features=config.hidden_dim,
#                 dropout=config.dropout,
#                 alpha=0.2  # LeakyReLU的负斜率
#             )
#             for i in range(config.num_layers)
#         ])
#
#         # 输出全连接层
#         self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
#         self.dropout = nn.Dropout(config.dropout)
#         self.fc2 = nn.Linear(config.hidden_dim, 1)
#
#         # 初始化参数
#         self._init_parameters()
#
#     def _init_parameters(self):
#         """初始化模型参数"""
#         # 初始化全连接层
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.zeros_(self.fc1.bias)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.zeros_(self.fc2.bias)
#
#     def forward(self, x, instruction_count=None):
#         """
#         前向传播
#
#         Args:
#             x: 输入数据 [batch_size, max_instr_count, max_instr_length]
#             instruction_count: 每个样本的指令数量 [batch_size]
#
#         Returns:
#             预测的吞吐量值 [batch_size]
#         """
#         batch_size, max_instr_count, max_instr_length = x.shape
#
#         # 生成指令级别掩码（屏蔽填充指令）
#         if instruction_count is not None:
#             instr_mask = torch.arange(max_instr_count, device=x.device)[None, :] >= instruction_count[:, None]
#         else:
#             instr_mask = torch.zeros((batch_size, max_instr_count), dtype=torch.bool, device=x.device)
#
#         # 词嵌入
#         token_embeds = self.token_embedding(x)  # [batch, instr_count, instr_len, embed_dim]
#
#         # 对每条指令进行平均池化，得到指令表示
#         padding_mask = (x != 0).float().unsqueeze(-1)  # [batch, instr_count, instr_len, 1]
#         masked_embeds = token_embeds * padding_mask
#         token_sum = masked_embeds.sum(dim=2)  # [batch, instr_count, embed_dim]
#         token_count = padding_mask.sum(dim=2)  # [batch, instr_count, 1]
#         token_count = torch.clamp(token_count, min=1.0)  # 避免除零
#         instr_repr = token_sum / token_count  # [batch, instr_count, embed_dim]
#
#         # 处理每个批次
#         outputs = []
#         for b in range(batch_size):
#             # 提取有效指令
#             valid_count = instruction_count[b].item() if instruction_count is not None else max_instr_count
#             valid_instrs = instr_repr[b, :valid_count]  # [valid_count, embed_dim]
#
#             # 如果没有有效指令，创建一个虚拟节点
#             if valid_count == 0:
#                 graph_repr = torch.zeros(self.config.hidden_dim, device=x.device)
#                 outputs.append(graph_repr)
#                 continue
#
#             # 构建全连接图的邻接矩阵（每条指令与其他所有指令相连）
#             adj_matrix = torch.ones(valid_count, valid_count, device=x.device) - torch.eye(valid_count, device=x.device)
#
#             # 应用GNN层
#             node_features = valid_instrs
#             for gnn_layer in self.gnn_layers:
#                 node_features = gnn_layer(node_features, adj_matrix)
#
#             # 全局池化
#             graph_repr = node_features.mean(dim=0)  # [hidden_dim]
#             outputs.append(graph_repr)
#
#         # 堆叠批次结果
#         out = torch.stack(outputs)  # [batch_size, hidden_dim]
#
#         # MLP预测
#         out = self.fc1(out)
#         out = F.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)
#
#         return out.squeeze(-1)  # [batch]
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import json

from .base_model import BaseModel
from .model import RISCVGraniteModel
from .graph_encoder import RISCVGraphEncoder


class GNNModel(BaseModel):
    """基于GRANITE的RISC-V指令集吞吐量预测GNN模型"""

    def __init__(self, config):
        """
        初始化GNN模型

        Args:
            config: 配置对象
        """
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

    # def forward(self, x, instruction_count=None):
    #     """
    #     前向传播
    #
    #     Args:
    #         x: 输入数据 [batch_size, max_instr_count, max_instr_length]
    #         instruction_count: 每个样本的指令数量 [batch_size]
    #
    #     Returns:
    #         预测的吞吐量值 [batch_size]
    #     """
    #     batch_size = x.size(0)
    #     results = []
    #
    #     # 由于图处理通常是单样本的，我们对每个样本分别处理
    #     for i in range(batch_size):
    #         # 确定有效的指令数量
    #         valid_count = instruction_count[i].item() if instruction_count is not None else x.size(1)
    #
    #         # 提取指令
    #         instructions = []
    #         for j in range(valid_count):
    #             # 从令牌ID列表构建指令字符串
    #             # 提取非零令牌
    #             tokens = [t.item() for t in x[i, j] if t.item() != 0]
    #             if tokens:
    #                 # 这里我们需要从令牌ID转换回实际的指令文本
    #                 instr_str = " ".join([str(t) for t in tokens])
    #                 instructions.append(instr_str)
    #
    #         if not instructions:
    #             # 如果没有有效指令，预测为0
    #             results.append(torch.tensor(0.0, device=x.device))
    #             continue
    #
    #         # 构建图
    #         graph = self.graph_encoder.build_graph(instructions)
    #
    #         # 将图移动到与x相同的设备
    #         graph = graph.to(x.device)
    #
    #         # 使用模型预测
    #         throughput = self.model(graph)
    #
    #         results.append(throughput)
    #
    #     # 堆叠所有结果
    #     return torch.stack(results)
    # def forward(self, x, instruction_count=None):
    #     """
    #     前向传播
    #
    #     Args:
    #         x: 输入数据 [batch_size, max_instr_count, max_instr_length]
    #         instruction_count: 每个样本的指令数量 [batch_size]
    #
    #     Returns:
    #         预测的吞吐量值 [batch_size]
    #     """
    #     batch_size = x.size(0)
    #     results = []
    #
    #     # 检查输入是否包含原始指令文本
    #     has_original_text = hasattr(x, 'instruction_text')
    #
    #     # 由于图处理通常是单样本的，我们对每个样本分别处理
    #     for i in range(batch_size):
    #         # 确定有效的指令数量
    #         valid_count = instruction_count[i].item() if instruction_count is not None else x.size(1)
    #
    #         if has_original_text:
    #             # 使用原始指令文本
    #             original_instrs = json.loads(x.instruction_text[i])
    #             instructions = original_instrs[:valid_count]
    #         else:
    #             # 从令牌ID列表构建指令字符串
    #             instructions = []
    #             for j in range(valid_count):
    #                 # 提取非零令牌
    #                 tokens = [t.item() for t in x[i, j] if t.item() != 0]
    #                 if tokens:
    #                     # 尝试将令牌ID转换回令牌文本
    #                     try:
    #                         token_strs = [self.inverse_vocab.get(t, f"<UNK:{t}>") for t in tokens]
    #                         instr_str = " ".join(token_strs)
    #                         instructions.append(instr_str)
    #                     except:
    #                         # 如果转换失败，使用占位符
    #                         instr_str = f"instr_{j}"
    #                         instructions.append(instr_str)
    #
    #         if not instructions:
    #             # 如果没有有效指令，预测为0
    #             results.append(torch.tensor(0.0, device=x.device))
    #             continue
    #
    #         # 构建图
    #         graph = self.graph_encoder.build_graph(instructions)
    #
    #         # 将图移动到与x相同的设备
    #         graph = graph.to(x.device)
    #
    #         # 使用模型预测
    #         throughput = self.model(graph)
    #
    #         results.append(throughput)
    #
    #     # 堆叠所有结果
    #     return torch.stack(results)

    def forward(self, x, instruction_count=None):
        """
        前向传播

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