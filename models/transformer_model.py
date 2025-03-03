# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, Optional
#
# from .base_model import BaseModel
#
# class TransformerModel(BaseModel):
#     """基于Transformer的RISC-V指令集吞吐量预测模型"""
#
#     def __init__(self, config):
#         """
#         初始化Transformer模型
#
#         Args:
#             config: 配置对象
#         """
#         super(TransformerModel, self).__init__(config)
#
#         # 嵌入层
#         self.token_embedding = nn.Embedding(
#             config.vocab_size,
#             config.embed_dim,
#             padding_idx=0
#         )
#
#         # 位置编码
#         self.position_embedding = nn.Parameter(
#             torch.zeros(1, config.max_instr_count, config.max_instr_length, config.embed_dim)
#         )
#
#         # Transformer编码器层
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=config.embed_dim,
#             nhead=config.num_heads,
#             dim_feedforward=config.hidden_dim,
#             dropout=config.dropout,
#             batch_first=True
#         )
#
#         # Transformer编码器
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=config.num_layers
#         )
#
#         # 输出全连接层
#         self.fc1 = nn.Linear(config.embed_dim, config.hidden_dim)
#         self.dropout = nn.Dropout(config.dropout)
#         self.fc2 = nn.Linear(config.hidden_dim, 1)
#
#         # 初始化参数
#         self._init_parameters()
#
#     def _init_parameters(self):
#         """初始化模型参数"""
#         # 初始化位置编码
#         nn.init.normal_(self.position_embedding, mean=0, std=0.02)
#
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
#         # 对每条指令进行编码
#         token_embeds = self.token_embedding(x)  # [batch, instr_count, instr_len, embed_dim]
#
#         # 添加位置编码
#         embeds = token_embeds + self.position_embedding[:, :max_instr_count, :max_instr_length, :]
#
#         # 重塑为[batch*instr_count, instr_len, embed_dim]以处理每条指令
#         embeds = embeds.reshape(-1, max_instr_length, self.config.embed_dim)
#
#         # 为transformer编码器生成注意力掩码（屏蔽填充token）
#         padding_mask = (x == 0).reshape(-1, max_instr_length)  # [batch*instr_count, instr_len]
#
#         # 使用transformer对每条指令进行编码
#         encoded = self.transformer_encoder(
#             embeds,
#             src_key_padding_mask=padding_mask
#         )  # [batch*instr_count, instr_len, embed_dim]
#
#         # 全局平均池化，忽略填充部分
#         # 先得到非填充部分的掩码
#         valid_token_mask = ~padding_mask.unsqueeze(-1)  # [batch*instr_count, instr_len, 1]
#
#         # 计算每个样本中有效token的数量
#         valid_token_counts = valid_token_mask.sum(dim=1)  # [batch*instr_count, 1]
#         valid_token_counts = torch.clamp(valid_token_counts, min=1.0)  # 避免除零
#
#         # 只考虑非填充部分的平均值
#         masked_encoded = encoded * valid_token_mask
#         pooled_instr = masked_encoded.sum(dim=1) / valid_token_counts  # [batch*instr_count, embed_dim]
#
#         # 重塑回[batch, instr_count, embed_dim]
#         pooled_instr = pooled_instr.reshape(batch_size, max_instr_count, self.config.embed_dim)
#
#         # 应用指令级别掩码
#         mask_expand = instr_mask.unsqueeze(-1).expand_as(pooled_instr)
#         pooled_instr = pooled_instr.masked_fill(mask_expand, 0)
#
#         # 对所有指令进行平均池化
#         valid_instr_counts = (~instr_mask).sum(dim=1, keepdim=True)
#         valid_instr_counts = torch.clamp(valid_instr_counts, min=1)  # 防止除零
#         pooled = pooled_instr.sum(dim=1) / valid_instr_counts  # [batch, embed_dim]
#
#         # MLP预测
#         out = self.fc1(pooled)  # [batch, hidden_dim]
#         out = F.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)  # [batch, 1]
#
#         return out.squeeze(-1)  # [batch]
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .base_model import BaseModel
from .DeepPM import DeepPM


class TransformerModel(BaseModel):
    """
    基于DeepPM的RISC-V指令集吞吐量预测模型
    """

    def __init__(self, config):
        """
        初始化Transformer模型

        Args:
            config: 配置对象
        """
        super(TransformerModel, self).__init__(config)

        # 创建DeepPM模型
        self.model = DeepPM(
            dim=config.embed_dim,
            n_heads=config.num_heads,
            dim_ff=config.hidden_dim,
            pad_idx=0,  # 假设0是填充索引
            vocab_size=config.vocab_size,
            num_basic_block_layer=2,
            num_instruction_layer=2,
            num_op_layer=config.num_layers,
            use_checkpoint=getattr(config, 'use_checkpoint', False),
            use_layernorm=getattr(config, 'use_layernorm', True),
            use_bb_attn=getattr(config, 'use_bb_attn', True),
            use_seq_attn=getattr(config, 'use_seq_attn', True),
            use_op_attn=getattr(config, 'use_op_attn', True),
            use_pos_2d=getattr(config, 'use_pos_2d', False),
            dropout=config.dropout,
            pred_drop=config.dropout,
            activation='gelu',
            handle_neg=getattr(config, 'handle_neg', False)
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
        # 创建填充掩码 (True表示要掩蔽的位置)
        mask = x == 0

        # 创建指令掩码
        if instruction_count is not None:
            instr_mask = torch.arange(x.size(1), device=x.device)[None, :] >= instruction_count[:, None]

            # 将超出指令数量的部分全部掩码
            for i in range(x.size(0)):  # 遍历批次
                if instruction_count[i] < x.size(1):
                    mask[i, instruction_count[i]:, :] = True

        # 准备模型输入
        model_input = {
            'x': x,
            'bb_attn_mod': None,  # 可选的注意力修饰符
            'seq_attn_mod': None,
            'op_attn_mod': None
        }

        # 前向传播
        output = self.model(model_input)

        return output