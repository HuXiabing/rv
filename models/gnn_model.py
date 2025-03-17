import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import re
import json
from typing import Dict, List, Tuple, Optional, Union

class GNNModel(nn.Module):

    def __init__(self, config):

        super(GNNModel, self).__init__()

        self.model = RISCVGraniteModel(
            node_embedding_dim=config.embed_dim,
            edge_embedding_dim=config.embed_dim,
            global_embedding_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_message_passing_steps=config.num_layers,
            dropout=config.dropout,
            use_layer_norm=getattr(config, 'use_layer_norm', True)
        )

    def count_parameters(self) -> int:

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch):
        output = self.model(batch)
        return output


class RISCVGraniteModel(nn.Module):

    def __init__(
            self,
            node_embedding_dim: int = 256,
            edge_embedding_dim: int = 256,
            global_embedding_dim: int = 256,
            hidden_dim: int = 256,
            num_message_passing_steps: int = 8,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
    ):

        super(RISCVGraniteModel, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.global_embedding_dim = global_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.use_layer_norm = use_layer_norm

        self.gnn = GraphNeuralNetwork(
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            global_embedding_dim=global_embedding_dim,
            hidden_dim=hidden_dim,
            num_message_passing_steps=num_message_passing_steps,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        self.decoder = ThroughputDecoder(
            node_embedding_dim=node_embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, basic_block_graph):

        node_embeddings, _, _ = self.gnn(
            basic_block_graph.x,
            basic_block_graph.edge_index,
            basic_block_graph.edge_attr,
            basic_block_graph.batch if hasattr(basic_block_graph, 'batch') else None
        )

        instruction_mask = basic_block_graph.instruction_mask
        instruction_embeddings = node_embeddings[instruction_mask]

        if hasattr(basic_block_graph, 'batch'):
            instruction_batch = basic_block_graph.batch[instruction_mask]
        else:
            instruction_batch = None

        output = self.decoder(instruction_embeddings, instruction_batch)

        return output


class GraphNeuralNetwork(nn.Module):

    def __init__(
            self,
            node_embedding_dim: int = 256,
            edge_embedding_dim: int = 256,
            global_embedding_dim: int = 256,
            hidden_dim: int = 256,
            num_message_passing_steps: int = 8,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
    ):

        super(GraphNeuralNetwork, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.global_embedding_dim = global_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.use_layer_norm = use_layer_norm

        # 节点类型和token嵌入
        self.node_type_embedding = nn.Embedding(10, node_embedding_dim // 2)  # 假设最多10种节点类型
        self.node_token_embedding = nn.Embedding(1000, node_embedding_dim // 2)  # 假设最多1000个token

        # 边类型嵌入
        self.edge_type_embedding = nn.Embedding(10, edge_embedding_dim)  # 假设最多10种边类型

        # 全局特征初始化
        self.global_init = nn.Linear(1, global_embedding_dim)

        # 消息传递层
        self.message_passing_layers = nn.ModuleList([
            MessagePassingLayer(
                node_embedding_dim=node_embedding_dim,
                edge_embedding_dim=edge_embedding_dim,
                global_embedding_dim=global_embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            ) for _ in range(num_message_passing_steps)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:

        node_type_emb = self.node_type_embedding(x[:, 0])
        node_token_emb = self.node_token_embedding(x[:, 1])
        node_embeddings = torch.cat([node_type_emb, node_token_emb], dim=1)

        edge_embeddings = self.edge_type_embedding(edge_attr.squeeze(-1))

        if batch is None:
            global_embedding = self.global_init(torch.ones(1, 1, device=x.device))
        else:
            num_graphs = batch.max().item() + 1
            global_embedding = self.global_init(torch.ones(num_graphs, 1, device=x.device))

        for i in range(self.num_message_passing_steps):
            node_embeddings, edge_embeddings, global_embedding = self.message_passing_layers[i](
                node_embeddings, edge_embeddings, global_embedding, edge_index, batch
            )

        return node_embeddings, edge_embeddings, global_embedding


class MessagePassingLayer(nn.Module):

    def __init__(
            self,
            node_embedding_dim: int,
            edge_embedding_dim: int,
            global_embedding_dim: int,
            hidden_dim: int,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
    ):

        super(MessagePassingLayer, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.global_embedding_dim = global_embedding_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

        # 边更新网络
        self.edge_update = nn.Sequential(
            nn.Linear(edge_embedding_dim + 2 * node_embedding_dim + global_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_embedding_dim)
        )

        # 节点更新网络
        self.node_update = nn.Sequential(
            nn.Linear(node_embedding_dim + hidden_dim + global_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_embedding_dim)
        )

        # 全局更新网络
        self.global_update = nn.Sequential(
            nn.Linear(global_embedding_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, global_embedding_dim)
        )

        # 层归一化
        if use_layer_norm:
            self.edge_layer_norm = nn.LayerNorm(edge_embedding_dim)
            self.node_layer_norm = nn.LayerNorm(node_embedding_dim)
            self.global_layer_norm = nn.LayerNorm(global_embedding_dim)

        # 用于消息聚合的投影矩阵
        self.edge_to_message = nn.Linear(edge_embedding_dim, hidden_dim)
        self.node_to_global = nn.Linear(node_embedding_dim, hidden_dim)
        self.edge_to_global = nn.Linear(edge_embedding_dim, hidden_dim)

    def forward(self, node_embeddings: torch.Tensor, edge_embeddings: torch.Tensor,
                global_embedding: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        num_nodes = node_embeddings.size(0)
        num_edges = edge_index.size(1)

        # 单个图的默认批次
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=node_embeddings.device)

        # 边更新
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        src_embeddings = node_embeddings[src_nodes]
        dst_embeddings = node_embeddings[dst_nodes]

        # 对于每条边，获取其图的全局嵌入
        edge_global_embeddings = global_embedding[batch[src_nodes]]

        # 连接源节点、目标节点、边和全局特征
        edge_inputs = torch.cat([
            src_embeddings,
            dst_embeddings,
            edge_embeddings,
            edge_global_embeddings
        ], dim=1)

        # 应用边更新网络
        edge_updates = self.edge_update(edge_inputs)

        # 应用残差连接和层归一化
        updated_edge_embeddings = edge_embeddings + edge_updates
        if self.use_layer_norm:
            updated_edge_embeddings = self.edge_layer_norm(updated_edge_embeddings)

        # 聚合每个节点的边消息
        # 将边嵌入投影为消息
        edge_messages = self.edge_to_message(updated_edge_embeddings)

        # 初始化节点消息
        node_messages = torch.zeros(num_nodes, self.hidden_dim, device=node_embeddings.device)

        # 聚合来自输入边的消息（到目标节点）
        for i in range(num_edges):
            node_messages[dst_nodes[i]] += edge_messages[i]

        # 节点更新
        # 对于每个节点，获取其图的全局嵌入
        node_global_embeddings = global_embedding[batch]

        # 连接节点特征、聚合的消息和全局特征
        node_inputs = torch.cat([
            node_embeddings,
            node_messages,
            node_global_embeddings
        ], dim=1)

        # 应用节点更新网络
        node_updates = self.node_update(node_inputs)

        # 应用残差连接和层归一化
        updated_node_embeddings = node_embeddings + node_updates
        if self.use_layer_norm:
            updated_node_embeddings = self.node_layer_norm(updated_node_embeddings)

        # 全局更新
        # 聚合用于全局更新的节点和边特征
        num_graphs = global_embedding.size(0)

        # 投影节点嵌入用于全局聚合
        node_features_for_global = self.node_to_global(updated_node_embeddings)

        # 投影边嵌入用于全局聚合
        edge_features_for_global = self.edge_to_global(updated_edge_embeddings)

        # 按图聚合节点嵌入
        node_aggregated = torch.zeros(num_graphs, self.hidden_dim, device=node_embeddings.device)
        for i in range(num_nodes):
            node_aggregated[batch[i]] += node_features_for_global[i]

        # 按图聚合边嵌入
        edge_aggregated = torch.zeros(num_graphs, self.hidden_dim, device=edge_embeddings.device)
        for i in range(num_edges):
            edge_aggregated[batch[src_nodes[i]]] += edge_features_for_global[i]

        # 连接全局特征和聚合的节点和边特征
        global_inputs = torch.cat([
            global_embedding,
            node_aggregated,
            edge_aggregated
        ], dim=1)

        # 应用全局更新网络
        global_updates = self.global_update(global_inputs)

        # 应用残差连接和层归一化
        updated_global_embedding = global_embedding + global_updates
        if self.use_layer_norm:
            updated_global_embedding = self.global_layer_norm(updated_global_embedding)

        return updated_node_embeddings, updated_edge_embeddings, updated_global_embedding

class ThroughputDecoder(nn.Module):

    def __init__(
            self,
            node_embedding_dim: int = 256,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
    ):
        super(ThroughputDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(node_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(node_embedding_dim)

    def forward(self, instruction_embeddings, batch=None):
        """
        解码器的前向传播

        Args:
            instruction_embeddings: 指令节点嵌入 [num_instructions, node_embedding_dim]
            batch: 指令节点的批次分配 [num_instructions]，指示每个指令属于哪个样本

        Returns:
            throughputs: 每个样本的预测吞吐量 [batch_size]
        """
        if self.use_layer_norm:
            instruction_embeddings = self.layer_norm(instruction_embeddings)

        # 计算每条指令的贡献
        instruction_contributions = self.decoder(instruction_embeddings).squeeze(-1)

        # 如果没有批次信息，则假设只有一个样本
        if batch is None:
            return torch.sum(instruction_contributions).unsqueeze(0)  # [1]

        # 按样本聚合指令贡献
        batch_size = batch.max().item() + 1
        throughputs = torch.zeros(batch_size, device=instruction_embeddings.device)

        # 使用scatter_add按批次聚合贡献
        throughputs.scatter_add_(0, batch, instruction_contributions)

        return throughputs  # [batch_size]