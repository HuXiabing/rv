import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network implementation following the GRANITE paper.
    Uses message passing to learn node, edge, and global embeddings.
    """
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
        
        # Node type and token embeddings
        self.node_type_embedding = nn.Embedding(10, node_embedding_dim // 2)  # Assuming at most 10 node types
        self.node_token_embedding = nn.Embedding(1000, node_embedding_dim // 2)  # Assuming at most 1000 tokens
        
        # Edge type embeddings
        self.edge_type_embedding = nn.Embedding(10, edge_embedding_dim)  # Assuming at most 10 edge types
        
        # Global feature initialization
        self.global_init = nn.Linear(1, global_embedding_dim)
        
        # Message passing layers
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
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass of the GNN.
        
        Args:
            x: Node features [num_nodes, 2] (type, token)
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, 1]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            node_embeddings: Updated node embeddings
            edge_embeddings: Updated edge embeddings
            global_embedding: Updated global embedding
        """
        # Initialize node embeddings
        node_type_emb = self.node_type_embedding(x[:, 0])
        node_token_emb = self.node_token_embedding(x[:, 1])
        node_embeddings = torch.cat([node_type_emb, node_token_emb], dim=1)
        
        # Initialize edge embeddings
        edge_embeddings = self.edge_type_embedding(edge_attr.squeeze(-1))
        
        # Initialize global features with a simple placeholder
        # In a real implementation, you might want to derive something meaningful from the graph
        if batch is None:
            # Single graph case
            global_embedding = self.global_init(torch.ones(1, 1, device=x.device))
        else:
            # Batched graphs case
            num_graphs = batch.max().item() + 1
            global_embedding = self.global_init(torch.ones(num_graphs, 1, device=x.device))
        
        # Apply message passing
        for i in range(self.num_message_passing_steps):
            node_embeddings, edge_embeddings, global_embedding = self.message_passing_layers[i](
                node_embeddings, edge_embeddings, global_embedding, edge_index, batch
            )
        
        return node_embeddings, edge_embeddings, global_embedding


class MessagePassingLayer(nn.Module):
    """
    A single message passing layer for the GNN, implementing the 'full GN block'
    architecture from the paper 'Relational inductive biases, deep learning, and graph networks'.
    """
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
        
        # Edge update network
        self.edge_update = nn.Sequential(
            nn.Linear(edge_embedding_dim + 2 * node_embedding_dim + global_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_embedding_dim)
        )
        
        # Node update network
        self.node_update = nn.Sequential(
            nn.Linear(node_embedding_dim + hidden_dim + global_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_embedding_dim)
        )
        
        # Global update network
        self.global_update = nn.Sequential(
            nn.Linear(global_embedding_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, global_embedding_dim)
        )
        
        # Layer normalization
        if use_layer_norm:
            self.edge_layer_norm = nn.LayerNorm(edge_embedding_dim)
            self.node_layer_norm = nn.LayerNorm(node_embedding_dim)
            self.global_layer_norm = nn.LayerNorm(global_embedding_dim)
        
        # Projection matrices for message aggregation
        self.edge_to_message = nn.Linear(edge_embedding_dim, hidden_dim)
        self.node_to_global = nn.Linear(node_embedding_dim, hidden_dim)
        self.edge_to_global = nn.Linear(edge_embedding_dim, hidden_dim)
    
    def forward(self, node_embeddings, edge_embeddings, global_embedding, edge_index, batch=None):
        """
        Forward pass of a single message passing layer.
        
        Args:
            node_embeddings: Current node embeddings [num_nodes, node_embedding_dim]
            edge_embeddings: Current edge embeddings [num_edges, edge_embedding_dim]
            global_embedding: Current global embedding [num_graphs, global_embedding_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
        
        Returns:
            updated_node_embeddings: Updated node embeddings
            updated_edge_embeddings: Updated edge embeddings
            updated_global_embedding: Updated global embedding
        """
        num_nodes = node_embeddings.size(0)
        num_edges = edge_index.size(1)
        
        # Default batch for a single graph
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=node_embeddings.device)
        
        # Edge update
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        src_embeddings = node_embeddings[src_nodes]
        dst_embeddings = node_embeddings[dst_nodes]
        
        # For each edge, get the global embedding of its graph
        edge_global_embeddings = global_embedding[batch[src_nodes]]
        
        # Concatenate source, destination, edge, and global features
        edge_inputs = torch.cat([
            src_embeddings, 
            dst_embeddings, 
            edge_embeddings, 
            edge_global_embeddings
        ], dim=1)
        
        # Apply edge update network
        edge_updates = self.edge_update(edge_inputs)
        
        # Apply residual connection and layer normalization
        updated_edge_embeddings = edge_embeddings + edge_updates
        if self.use_layer_norm:
            updated_edge_embeddings = self.edge_layer_norm(updated_edge_embeddings)
        
        # Aggregate edge messages for each node
        # Project edge embeddings to messages
        edge_messages = self.edge_to_message(updated_edge_embeddings)
        
        # Initialize node messages
        node_messages = torch.zeros(num_nodes, self.hidden_dim, device=node_embeddings.device)
        
        # Aggregate messages from incoming edges (to the destination nodes)
        for i in range(num_edges):
            node_messages[dst_nodes[i]] += edge_messages[i]
        
        # Node update
        # For each node, get the global embedding of its graph
        node_global_embeddings = global_embedding[batch]
        
        # Concatenate node features, aggregated messages, and global features
        node_inputs = torch.cat([
            node_embeddings, 
            node_messages, 
            node_global_embeddings
        ], dim=1)
        
        # Apply node update network
        node_updates = self.node_update(node_inputs)
        
        # Apply residual connection and layer normalization
        updated_node_embeddings = node_embeddings + node_updates
        if self.use_layer_norm:
            updated_node_embeddings = self.node_layer_norm(updated_node_embeddings)
        
        # Global update
        # Aggregate node and edge features for global update
        num_graphs = global_embedding.size(0)
        
        # Project node embeddings for global aggregation
        node_features_for_global = self.node_to_global(updated_node_embeddings)
        
        # Project edge embeddings for global aggregation
        edge_features_for_global = self.edge_to_global(updated_edge_embeddings)
        
        # Aggregate node embeddings per graph
        node_aggregated = torch.zeros(num_graphs, self.hidden_dim, device=node_embeddings.device)
        for i in range(num_nodes):
            node_aggregated[batch[i]] += node_features_for_global[i]
        
        # Aggregate edge embeddings per graph
        edge_aggregated = torch.zeros(num_graphs, self.hidden_dim, device=edge_embeddings.device)
        for i in range(num_edges):
            edge_aggregated[batch[src_nodes[i]]] += edge_features_for_global[i]
        
        # Concatenate global features and aggregated node and edge features
        global_inputs = torch.cat([
            global_embedding, 
            node_aggregated, 
            edge_aggregated
        ], dim=1)
        
        # Apply global update network
        global_updates = self.global_update(global_inputs)
        
        # Apply residual connection and layer normalization
        updated_global_embedding = global_embedding + global_updates
        if self.use_layer_norm:
            updated_global_embedding = self.global_layer_norm(updated_global_embedding)
        
        return updated_node_embeddings, updated_edge_embeddings, updated_global_embedding
