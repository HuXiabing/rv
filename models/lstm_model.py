# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, Optional
#
# from .base_model import BaseModel
#
# class LSTMModel(BaseModel):
#     """基于LSTM的RISC-V指令集吞吐量预测模型"""
#
#     def __init__(self, config):
#         """
#         初始化LSTM模型
#
#         Args:
#             config: 配置对象
#         """
#         super(LSTMModel, self).__init__(config)
#
#         # 嵌入层
#         self.token_embedding = nn.Embedding(
#             config.vocab_size,
#             config.embed_dim,
#             padding_idx=0
#         )
#
#         # 指令级别LSTM
#         self.instr_lstm = nn.LSTM(
#             input_size=config.embed_dim,
#             hidden_size=config.hidden_dim // 2,  # 双向LSTM，所以隐藏维度减半
#             num_layers=config.num_layers // 2,
#             dropout=config.dropout if config.num_layers > 2 else 0,
#             batch_first=True,
#             bidirectional=True
#         )
#
#         # 序列级别LSTM
#         self.seq_lstm = nn.LSTM(
#             input_size=config.hidden_dim,  # 上一层双向LSTM的输出
#             hidden_size=config.hidden_dim // 2,  # 双向LSTM，所以隐藏维度减半
#             num_layers=config.num_layers // 2,
#             dropout=config.dropout if config.num_layers > 2 else 0,
#             batch_first=True,
#             bidirectional=True
#         )
#
#         # 输出全连接层
#         self.fc = nn.Linear(config.hidden_dim, 1)
#
#         # 初始化参数
#         self._init_parameters()
#
#     def _init_parameters(self):
#         """初始化模型参数"""
#         # 初始化LSTM参数
#         for name, param in self.instr_lstm.named_parameters():
#             if 'weight' in name:
#                 nn.init.orthogonal_(param)
#             elif 'bias' in name:
#                 nn.init.zeros_(param)
#
#         for name, param in self.seq_lstm.named_parameters():
#             if 'weight' in name:
#                 nn.init.orthogonal_(param)
#             elif 'bias' in name:
#                 nn.init.zeros_(param)
#
#         # 初始化全连接层
#         nn.init.xavier_uniform_(self.fc.weight)
#         nn.init.zeros_(self.fc.bias)
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
#         # 重塑为[batch*instr_count, instr_len, embed_dim]以处理每条指令
#         token_embeds = token_embeds.reshape(-1, max_instr_length, self.config.embed_dim)
#
#         # 对每条指令应用LSTM
#         instr_out, _ = self.instr_lstm(token_embeds)  # [batch*instr_count, instr_len, hidden_dim]
#
#         # 获取最后一个非填充位置的输出作为指令表示
#         # 创建掩码，标记出非填充位置
#         padding_mask = (x != 0).long()  # [batch, instr_count, instr_len]
#         padding_mask = padding_mask.reshape(-1, max_instr_length)  # [batch*instr_count, instr_len]
#
#         # 获取每个序列的最后一个非填充位置的索引
#         seq_lengths = padding_mask.sum(dim=1) - 1  # [batch*instr_count]
#         seq_lengths = torch.clamp(seq_lengths, min=0)  # 处理全是填充的情况
#
#         # 获取对应位置的hidden state
#         batch_indices = torch.arange(batch_size * max_instr_count, device=x.device)
#         last_states = instr_out[batch_indices, seq_lengths]  # [batch*instr_count, hidden_dim]
#
#         # 重塑回[batch, instr_count, hidden_dim]
#         instr_repr = last_states.reshape(batch_size, max_instr_count, -1)  # [batch, instr_count, hidden_dim]
#
#         # 应用指令掩码，将填充指令的表示设为0
#         mask_expand = instr_mask.unsqueeze(-1).expand_as(instr_repr)
#         instr_repr = instr_repr.masked_fill(mask_expand, 0)
#
#         # 对指令序列应用LSTM
#         seq_out, _ = self.seq_lstm(instr_repr)  # [batch, instr_count, hidden_dim]
#
#         # 获取有效指令的数量
#         valid_instr_counts = (~instr_mask).sum(dim=1) - 1  # [batch]
#         valid_instr_counts = torch.clamp(valid_instr_counts, min=0)  # 处理没有有效指令的情况
#
#         # 获取每个样本的最后一个有效指令的序列输出
#         batch_indices = torch.arange(batch_size, device=x.device)
#         last_seq_states = seq_out[batch_indices, valid_instr_counts]  # [batch, hidden_dim]
#
#         # 全连接层预测
#         out = self.fc(last_seq_states)  # [batch, 1]
#
#         return out.squeeze(-1)  # [batch]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .base_model import BaseModel
from .Ithemal import BatchRNN


class LSTMModel(BaseModel):
    """基于Ithemal的RISC-V指令集吞吐量预测模型，采用BatchRNN实现"""

    def __init__(self, config):
        """
        初始化LSTM模型

        Args:
            config: 配置对象
        """
        super(LSTMModel, self).__init__(config)

        # 创建BatchRNN模型
        self.model = BatchRNN(
            embedding_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_classes=1,  # 回归任务
            pad_idx=0,  # 假设0是填充索引
            num_layers=config.num_layers,
            vocab_size=config.vocab_size
        )

    def forward(self, x, instruction_count=None):
        """
        前向传播

        Args:
            x: 输入数据 [batch_size, max_instr_count, max_instr_length]
            instruction_count: 每个样本的指令数量 [batch_size]

        Returns:
            预测的吞吐量值 [batch_size]
        """
        # BatchRNN已经设计为处理批处理输入
        # 所以可以直接传递x
        return self.model(x)