import torch
import torch.nn as nn
import torch.nn.functional as F

class ThroughputDecoder(nn.Module):
    """
    Decoder network to predict throughput from instruction embeddings.
    Following the GRANITE model, it computes the contribution of each
    instruction to the overall throughput and then sums them.
    """
    def __init__(
        self,
        node_embedding_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super(ThroughputDecoder, self).__init__()
        
        # Two-layer feed-forward ReLU network with residual connections
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
    
    def forward(self, instruction_embeddings):
        """
        Forward pass of the decoder.
        
        Args:
            instruction_embeddings: Embeddings of instruction nodes [num_instructions, node_embedding_dim]
            
        Returns:
            throughput: Predicted throughput of the basic block
        """
        if self.use_layer_norm:
            instruction_embeddings = self.layer_norm(instruction_embeddings)
        
        # Apply decoder to each instruction embedding to get per-instruction contribution
        instruction_contributions = self.decoder(instruction_embeddings)
        
        # Sum all instruction contributions to get the final throughput
        throughput = torch.sum(instruction_contributions)
        
        return throughput


class MultiTaskThroughputDecoder(nn.Module):
    """
    Multi-task version of the throughput decoder.
    Has separate decoder networks for each microarchitecture.
    """
    def __init__(
        self,
        node_embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_tasks: int = 3,  # Number of microarchitectures
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super(MultiTaskThroughputDecoder, self).__init__()
        
        # Create a separate decoder for each task (microarchitecture)
        self.decoders = nn.ModuleList([
            ThroughputDecoder(
                node_embedding_dim=node_embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            ) for _ in range(num_tasks)
        ])
        
        self.num_tasks = num_tasks
    
    def forward(self, instruction_embeddings, task_id=0):
        """
        Forward pass of the multi-task decoder.
        
        Args:
            instruction_embeddings: Embeddings of instruction nodes [num_instructions, node_embedding_dim]
            task_id: ID of the task (microarchitecture) to use for decoding
            
        Returns:
            throughput: Predicted throughput of the basic block for the specified task
        """
        if task_id >= self.num_tasks:
            raise ValueError(f"Task ID {task_id} is out of range (max: {self.num_tasks-1})")
        
        # Use the decoder for the specified task
        return self.decoders[task_id](instruction_embeddings)
    
    def forward_all(self, instruction_embeddings):
        """
        Forward pass of the multi-task decoder for all tasks.
        
        Args:
            instruction_embeddings: Embeddings of instruction nodes [num_instructions, node_embedding_dim]
            
        Returns:
            throughputs: Predicted throughputs of the basic block for all tasks
        """
        throughputs = []
        for i in range(self.num_tasks):
            throughputs.append(self.decoders[i](instruction_embeddings))
        
        return torch.stack(throughputs)
