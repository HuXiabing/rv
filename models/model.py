import torch
import torch.nn as nn
from .granite_gnn import GraphNeuralNetwork
from .decoder import ThroughputDecoder, MultiTaskThroughputDecoder

class RISCVGraniteModel(nn.Module):
    """
    A GNN-based model for RISC-V basic block throughput estimation,
    inspired by the GRANITE model from the paper.
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
        num_tasks: int = 1,  # For multi-task learning (multiple microarchitectures)
        use_multi_task_decoder: bool = False,  # Whether to use a multi-task decoder
    ):
        super(RISCVGraniteModel, self).__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.global_embedding_dim = global_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.use_layer_norm = use_layer_norm
        self.num_tasks = num_tasks
        self.use_multi_task_decoder = use_multi_task_decoder
        
        # Initialize the GNN
        self.gnn = GraphNeuralNetwork(
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            global_embedding_dim=global_embedding_dim,
            hidden_dim=hidden_dim,
            num_message_passing_steps=num_message_passing_steps,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        
        # Initialize the decoder(s)
        if use_multi_task_decoder:
            self.decoder = MultiTaskThroughputDecoder(
                node_embedding_dim=node_embedding_dim,
                hidden_dim=hidden_dim,
                num_tasks=num_tasks,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
        else:
            # Initialize decoders for each task (microarchitecture)
            self.decoders = nn.ModuleList([
                ThroughputDecoder(
                    node_embedding_dim=node_embedding_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                ) for _ in range(num_tasks)
            ])
    
    def forward(self, basic_block_graph, task_id=0):
        """
        Forward pass of the model.
        
        Args:
            basic_block_graph: The graph representation of a RISC-V basic block
            task_id: ID of the task (microarchitecture) to use for decoding
            
        Returns:
            throughput: Estimated throughput of the basic block
        """
        # Apply the GNN to get node embeddings
        node_embeddings, _, _ = self.gnn(
            basic_block_graph.x,
            basic_block_graph.edge_index,
            basic_block_graph.edge_attr,
            basic_block_graph.batch if hasattr(basic_block_graph, 'batch') else None
        )
        
        # Get instruction nodes (using the instruction mask)
        instruction_mask = basic_block_graph.instruction_mask
        instruction_embeddings = node_embeddings[instruction_mask]
        
        # Apply the decoder to get throughput estimation
        if self.use_multi_task_decoder:
            throughput = self.decoder(instruction_embeddings, task_id)
        else:
            # Select the appropriate decoder based on task_id
            decoder = self.decoders[task_id]
            throughput = decoder(instruction_embeddings)
        
        return throughput
    
    def forward_all_tasks(self, basic_block_graph):
        """
        Forward pass of the model for all tasks.
        
        Args:
            basic_block_graph: The graph representation of a RISC-V basic block
            
        Returns:
            throughputs: Estimated throughputs of the basic block for all tasks
        """
        # Apply the GNN to get node embeddings
        node_embeddings, _, _ = self.gnn(
            basic_block_graph.x,
            basic_block_graph.edge_index,
            basic_block_graph.edge_attr,
            basic_block_graph.batch if hasattr(basic_block_graph, 'batch') else None
        )
        
        # Get instruction nodes (using the instruction mask)
        instruction_mask = basic_block_graph.instruction_mask
        instruction_embeddings = node_embeddings[instruction_mask]
        
        # Apply the decoder to get throughput estimations for all tasks
        if self.use_multi_task_decoder:
            throughputs = self.decoder.forward_all(instruction_embeddings)
        else:
            throughputs = []
            for i in range(self.num_tasks):
                throughputs.append(self.decoders[i](instruction_embeddings))
            throughputs = torch.stack(throughputs)
        
        return throughputs
