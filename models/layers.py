import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float, alpha: float = 0.2, concat: bool = True):
        """
        初始化图注意力层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: Dropout概率
            alpha: LeakyReLU的负斜率
            concat: 是否在多头注意力中使用拼接（最后一层通常为False）
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 线性变换矩阵
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # 注意力向量
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(alpha)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征矩阵，形状为 [num_nodes, in_features]
            adj: 邻接矩阵，形状为 [num_nodes, num_nodes]
            
        Returns:
            更新后的节点特征矩阵，形状为 [num_nodes, out_features]
        """
        # 线性变换
        Wh = self.W(x)  # [num_nodes, out_features]
        
        # 计算注意力系数
        num_nodes = Wh.size(0)
        
        # 为每对节点拼接特征
        a_input = self._prepare_attentional_mechanism_input(Wh)  # [num_nodes, num_nodes, 2*out_features]
        
        # 应用注意力向量
        e = torch.matmul(a_input, self.a).squeeze(-1)  # [num_nodes, num_nodes]
        
        # 应用LeakyReLU
        e = self.leakyrelu(e)
        
        # 屏蔽没有连接的节点对
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # 归一化注意力权重
        attention = F.softmax(attention, dim=1)
        
        # Dropout
        attention = self.dropout_layer(attention)
        
        # 加权聚合邻居的特征
        h_prime = torch.matmul(attention, Wh)  # [num_nodes, out_features]
        
        # 根据是否拼接选择激活函数
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """
        准备注意力机制的输入
        
        Args:
            Wh: 经过线性变换的节点特征，形状为 [num_nodes, out_features]
            
        Returns:
            注意力机制的输入张量，形状为 [num_nodes, num_nodes, 2*out_features]
        """
        num_nodes = Wh.size(0)
        
        # 复制节点特征用于拼接
        # [num_nodes, 1, out_features] 和 [1, num_nodes, out_features]
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=0).view(num_nodes, num_nodes, self.out_features)
        Wh_repeated_alternating = Wh.repeat(num_nodes, 1)
        Wh_repeated_alternating = Wh_repeated_alternating.view(num_nodes, num_nodes, self.out_features)
        
        # 拼接
        all_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)  # [num_nodes, num_nodes, 2*out_features]
        
        return all_combinations
