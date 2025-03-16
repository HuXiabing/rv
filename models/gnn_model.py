import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import re
import json
from typing import Dict, List, Tuple, Optional, Union
#
# class GNNModel(nn.Module):
#     """
#     Graph Neural Network model for RISC-V throughput prediction.
#
#     This model combines a graph encoder and GNN model to predict throughput
#     for RISC-V basic blocks.
#     """
#
#     def __init__(self, config):
#         """
#         Initialize the GNN model.
#
#         Args:
#             config: Configuration object with model hyperparameters
#         """
#         super(GNNModel, self).__init__()
#
#         # Create the graph encoder
#         self.graph_encoder = RISCVGraphEncoder()
#
#         # Create the RISC-V GRANITE model
#         self.model = RISCVGraniteModel(
#             node_embedding_dim=config.embed_dim,
#             edge_embedding_dim=config.embed_dim,
#             global_embedding_dim=config.embed_dim,
#             hidden_dim=config.hidden_dim,
#             num_message_passing_steps=config.num_layers,
#             dropout=config.dropout,
#             use_layer_norm=getattr(config, 'use_layer_norm', True)
#         )
#
#     def count_parameters(self) -> int:
#
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)
#
#     def _convert_to_basic_block(self, x: torch.Tensor, instruction_count: Optional[torch.Tensor]) -> List[List[str]]:
#         """
#         Convert model inputs to RISC-V basic block format.
#
#         Args:
#             x: Input tensor [batch_size, max_instr_count, max_instr_length]
#             instruction_count: Instruction count tensor [batch_size]
#
#         Returns:
#             List of RISC-V basic block instruction lists
#         """
#         batch_size = x.size(0)
#         basic_blocks = []
#
#         for i in range(batch_size):
#             # Determine valid instruction count
#             valid_count = instruction_count[i].item() if instruction_count is not None else x.size(1)
#
#             # Extract and convert instructions
#             instructions = []
#             for j in range(valid_count):
#                 tokens = [t.item() for t in x[i, j] if t.item() != 0]
#                 if tokens:
#                     instr_str = f"instr_{j}"
#                     instructions.append(instr_str)
#
#             basic_blocks.append(instructions)
#
#         return basic_blocks
#
#     def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]],
#                 instruction_count: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Forward pass of the model.
#
#         Args:
#             x: Input data (tensor or batch dictionary)
#             instruction_count: Optional tensor with instruction counts
#
#         Returns:
#             Predicted throughput values [batch_size]
#         """
#         # Handle input as dictionary from dataloader
#         if isinstance(x, dict):
#             batch_size = x['X'].size(0)
#             device = x['X'].device
#
#             if 'instruction_text' in x:
#                 instruction_texts = x['instruction_text']
#                 if instruction_count is None and 'instruction_count' in x:
#                     instruction_count = x['instruction_count']
#             else:
#                 instruction_texts = None
#                 x_tensor = x['X']
#                 if instruction_count is None and 'instruction_count' in x:
#                     instruction_count = x['instruction_count']
#         else:
#             # Direct tensor input
#             batch_size = x.size(0)
#             device = x.device
#             instruction_texts = None
#             x_tensor = x
#
#         results = []
#
#         # Process each sample in the batch
#         for i in range(batch_size):
#             # Determine valid instruction count
#             valid_count = instruction_count[i].item() if instruction_count is not None else x_tensor.size(1)
#
#             # Get instruction text
#             if instruction_texts is not None:
#                 instructions = json.loads(instruction_texts[i])[:valid_count]
#             else:
#                 instructions = []
#                 for j in range(valid_count):
#                     tokens = [t.item() for t in x_tensor[i, j] if t.item() != 0]
#                     if tokens:
#                         instr_str = " ".join([str(t) for t in tokens])
#                         instructions.append(instr_str)
#
#             if not instructions:
#                 # No valid instructions, predict 0
#                 results.append(torch.tensor(0.0, device=device))
#                 continue
#
#             # Build graph representation
#             graph = self.graph_encoder.build_graph(instructions)
#
#             # Move graph to the same device
#             graph = graph.to(device)
#
#             # Predict throughput
#             throughput = self.model(graph)
#
#             results.append(throughput)
#
#         # Stack all results
#         return torch.stack(results)
#
#
# class RISCVGraniteModel(nn.Module):
#     """
#     A GNN-based model for RISC-V basic block throughput estimation,
#     inspired by the GRANITE model from the paper.
#     """
#
#     def __init__(
#             self,
#             node_embedding_dim: int = 256,
#             edge_embedding_dim: int = 256,
#             global_embedding_dim: int = 256,
#             hidden_dim: int = 256,
#             num_message_passing_steps: int = 8,
#             dropout: float = 0.1,
#             use_layer_norm: bool = True,
#     ):
#         """
#         Initialize the RISC-V GRANITE model.
#
#         Args:
#             node_embedding_dim: Dimension of node embeddings
#             edge_embedding_dim: Dimension of edge embeddings
#             global_embedding_dim: Dimension of global embeddings
#             hidden_dim: Dimension of hidden layers
#             num_message_passing_steps: Number of message passing iterations
#             dropout: Dropout rate
#             use_layer_norm: Whether to use layer normalization
#         """
#         super(RISCVGraniteModel, self).__init__()
#
#         self.node_embedding_dim = node_embedding_dim
#         self.edge_embedding_dim = edge_embedding_dim
#         self.global_embedding_dim = global_embedding_dim
#         self.hidden_dim = hidden_dim
#         self.num_message_passing_steps = num_message_passing_steps
#         self.use_layer_norm = use_layer_norm
#
#         # Initialize the GNN
#         self.gnn = GraphNeuralNetwork(
#             node_embedding_dim=node_embedding_dim,
#             edge_embedding_dim=edge_embedding_dim,
#             global_embedding_dim=global_embedding_dim,
#             hidden_dim=hidden_dim,
#             num_message_passing_steps=num_message_passing_steps,
#             dropout=dropout,
#             use_layer_norm=use_layer_norm,
#         )
#
#         # Initialize the decoder
#         self.decoder = ThroughputDecoder(
#             node_embedding_dim=node_embedding_dim,
#             hidden_dim=hidden_dim,
#             dropout=dropout,
#             use_layer_norm=use_layer_norm,
#         )
#
#     def forward(self, basic_block_graph: torch_geometric.data.Data) -> torch.Tensor:
#         """
#         Forward pass of the model.
#
#         Args:
#             basic_block_graph: Graph representation of a RISC-V basic block
#
#         Returns:
#             throughput: Estimated throughput of the basic block
#         """
#         # Apply the GNN to get node embeddings
#         node_embeddings, _, _ = self.gnn(
#             basic_block_graph.x,
#             basic_block_graph.edge_index,
#             basic_block_graph.edge_attr,
#             basic_block_graph.batch if hasattr(basic_block_graph, 'batch') else None
#         )
#
#         # Get instruction nodes using the instruction mask
#         instruction_mask = basic_block_graph.instruction_mask
#         instruction_embeddings = node_embeddings[instruction_mask]
#
#         # Apply the decoder to get throughput estimation
#         throughput = self.decoder(instruction_embeddings)
#
#         return throughput
#
#
# class RISCVGraphEncoder:
#     """
#     Encodes RISC-V basic blocks into graph representations.
#     """
#
#     def __init__(self):
#         """Initialize the RISC-V graph encoder with necessary mappings."""
#         # Define node types
#         self.node_types = {
#             'mnemonic': 0,  # Instruction mnemonic (e.g., 'addi')
#             'register': 1,  # Register (e.g., 'x1')
#             'immediate': 2,  # Immediate value
#             'memory': 3,  # Memory value
#             'address': 4,  # Address computation
#             'prefix': 5,  # Instruction prefix
#         }
#
#         # Define edge types
#         self.edge_types = {
#             'structural': 0,  # From one instruction to the next
#             'input': 1,  # From value node to instruction node
#             'output': 2,  # From instruction node to value node
#             'address_base': 3,  # From register to address
#             'address_offset': 4,  # From immediate to address
#         }
#
#         # Define token to index mappings
#         self.token_to_idx = {'<UNK>': 0}
#         self.edge_type_to_idx = {}
#
#         # RISC-V instruction mnemonics
#         risc_v_instructions = [
#             "add", "addi", "addiw", "addw", "and", "andi", "auipc", "beq", "bge",
#             "bgeu", "blt", "bltu", "bne", "ebreak", "ecall", "fence", "jal", "jalr",
#             "lb", "lbu", "ld", "lh", "lhu", "lui", "lw", "lwu", "or", "ori", "sb",
#             "sd", "sh", "sll", "slli", "slliw", "sllw", "slt", "slti", "sltiu",
#             "sltu", "sra", "srai", "sraiw", "sraw", "srl", "srli", "srliw", "srlw",
#             "sub", "subw", "sw", "xor", "xori",
#             "amoadd.d", "amoadd.w", "amoand.d", "amoand.w", "amomax.d", "amomax.w",
#             "amomaxu.d", "amomaxu.w", "amomin.d", "amomin.w", "amominu.d", "amominu.w",
#             "amoor.d", "amoor.w", "amoswap.d", "amoswap.w", "amoxor.d", "amoxor.w",
#             "lr.d", "lr.w", "sc.d", "sc.w",
#             "div", "divu", "divuw", "divw", "mul", "mulh", "mulhsu", "mulhu", "mulw",
#             "rem", "remu", "remuw", "remw",
#             "fadd.d", "fadd.s", "fclass.d", "fclass.s", "fcvt.d.l", "fcvt.d.lu", "fcvt.d.s",
#             "fcvt.d.w", "fcvt.d.wu", "fcvt.l.d", "fcvt.l.s", "fcvt.lu.d", "fcvt.lu.s",
#             "fcvt.s.d", "fcvt.s.l", "fcvt.s.lu", "fcvt.s.w", "fcvt.s.wu", "fcvt.w.d",
#             "fcvt.w.s", "fcvt.wu.d", "fcvt.wu.s", "fdiv.d", "fdiv.s", "feq.d", "feq.s",
#             "fld", "fle.d", "fle.s", "flt.d", "flt.s", "flw", "fmadd.d", "fmadd.s",
#             "fmax.d", "fmax.s", "fmin.d", "fmin.s", "fmsub.d", "fmsub.s", "fmul.d",
#             "fmul.s", "fmv.d.x", "fmv.w.x", "fmv.x.d", "fmv.x.w", "fnmadd.d", "fnmadd.s",
#             "fnmsub.d", "fnmsub.s", "fsd", "fsgnj.d", "fsgnj.s", "fsgnjn.d", "fsgnjn.s",
#             "fsgnjx.d", "fsgnjx.s", "fsqrt.d", "fsqrt.s", "fsub.d", "fsub.s", "fsw"
#         ]
#
#         # Add RISC-V instructions to token mapping
#         for i, instr in enumerate(risc_v_instructions):
#             self.token_to_idx[instr] = i + 1  # +1 because 0 is <UNK>
#
#         # RISC-V registers (x0-x31, plus aliases)
#         registers = [f'x{i}' for i in range(32)]
#         register_aliases = {
#             'zero': 'x0', 'ra': 'x1', 'sp': 'x2', 'gp': 'x3', 'tp': 'x4',
#             't0': 'x5', 't1': 'x6', 't2': 'x7', 's0': 'x8', 'fp': 'x8',
#             's1': 'x9', 'a0': 'x10', 'a1': 'x11', 'a2': 'x12', 'a3': 'x13',
#             'a4': 'x14', 'a5': 'x15', 'a6': 'x16', 'a7': 'x17',
#             's2': 'x18', 's3': 'x19', 's4': 'x20', 's5': 'x21', 's6': 'x22',
#             's7': 'x23', 's8': 'x24', 's9': 'x25', 's10': 'x26', 's11': 'x27',
#             't3': 'x28', 't4': 'x29', 't5': 'x30', 't6': 'x31'
#         }
#
#         # Add RISC-V registers to token mapping
#         next_idx = len(self.token_to_idx)
#         for i, reg in enumerate(registers):
#             self.token_to_idx[reg] = next_idx + i
#
#         # Add register aliases
#         for alias, reg in register_aliases.items():
#             self.token_to_idx[alias] = self.token_to_idx[reg]
#
#         # Special tokens for immediate values, memory, and address computation
#         self.token_to_idx['<IMM>'] = len(self.token_to_idx)
#         self.token_to_idx['<MEM>'] = len(self.token_to_idx)
#         self.token_to_idx['<ADDR>'] = len(self.token_to_idx)
#
#         # Map edge types to indices
#         for i, edge_type in enumerate(self.edge_types.keys()):
#             self.edge_type_to_idx[edge_type] = i
#
#     def parse_instruction(self, instruction: str) -> Dict:
#         """
#         Parse a RISC-V instruction into its components.
#
#         Args:
#             instruction: A RISC-V assembly instruction string
#
#         Returns:
#             A dictionary with instruction components
#         """
#         instruction = instruction.strip().lower()
#
#         # Simple regex to extract mnemonic and operands
#         match = re.match(r'([a-z0-9\.]+)\s*(.*)', instruction)
#         if not match:
#             return {'mnemonic': '<UNK>', 'operands': []}
#
#         mnemonic, operands_str = match.groups()
#
#         # Split operands by commas, handling potential spaces
#         operands = [op.strip() for op in operands_str.split(',')] if operands_str else []
#
#         return {
#             'mnemonic': mnemonic,
#             'operands': operands
#         }
#
#     def build_graph(self, basic_block: List[str]) -> torch_geometric.data.Data:
#         """
#         Build a graph representation of a RISC-V basic block.
#
#         Args:
#             basic_block: List of RISC-V assembly instructions
#
#         Returns:
#             A PyTorch Geometric Data object representing the graph
#         """
#         nodes = []  # (type, token)
#         edges = []  # (src, dst, type)
#
#         # Maps to track nodes for values (registers, memory, etc.)
#         value_nodes = {}  # Maps value name to node index
#         instruction_nodes = []  # List of instruction node indices
#
#         node_idx = 0
#
#         for instr_idx, instruction in enumerate(basic_block):
#             parsed = self.parse_instruction(instruction)
#             mnemonic = parsed['mnemonic']
#             operands = parsed['operands']
#
#             # Add instruction mnemonic node
#             mnemonic_node_idx = node_idx
#             nodes.append((self.node_types['mnemonic'], self.token_to_idx.get(mnemonic, 0)))
#             instruction_nodes.append(mnemonic_node_idx)
#             node_idx += 1
#
#             # Add structural dependency edge to previous instruction
#             if instr_idx > 0:
#                 edges.append((instruction_nodes[instr_idx - 1], mnemonic_node_idx, self.edge_types['structural']))
#
#             # Process destination operands (outputs)
#             if operands:
#                 dest_operand = operands[0]
#
#                 # Check if it's a register
#                 if dest_operand in self.token_to_idx and dest_operand.startswith(
#                         ('x', 'a', 's', 't', 'ra', 'sp', 'gp', 'tp', 'fp', 'zero')):
#                     # Create a new register node for the destination
#                     dest_node_idx = node_idx
#                     nodes.append((self.node_types['register'], self.token_to_idx.get(dest_operand, 0)))
#                     node_idx += 1
#
#                     # Add output edge from instruction to register
#                     edges.append((mnemonic_node_idx, dest_node_idx, self.edge_types['output']))
#
#                     # Update value_nodes to point to this new node
#                     value_nodes[dest_operand] = dest_node_idx
#
#                 # Check if it's a memory store
#                 elif '(' in dest_operand and ')' in dest_operand:
#                     # Memory store like "sw x1, 8(x2)"
#                     # Extract offset and base register
#                     match = re.match(r'(\d+)\(([^\)]+)\)', dest_operand)
#                     if match:
#                         offset, base_reg = match.groups()
#
#                         # Create address node
#                         addr_node_idx = node_idx
#                         nodes.append((self.node_types['address'], self.token_to_idx['<ADDR>']))
#                         node_idx += 1
#
#                         # Create memory node
#                         mem_node_idx = node_idx
#                         nodes.append((self.node_types['memory'], self.token_to_idx['<MEM>']))
#                         node_idx += 1
#
#                         # Add base register to address edge
#                         if base_reg in value_nodes:
#                             edges.append((value_nodes[base_reg], addr_node_idx, self.edge_types['address_base']))
#
#                         # Add immediate offset to address edge
#                         imm_node_idx = node_idx
#                         nodes.append((self.node_types['immediate'], self.token_to_idx['<IMM>']))
#                         node_idx += 1
#                         edges.append((imm_node_idx, addr_node_idx, self.edge_types['address_offset']))
#
#                         # Add output edge from instruction to memory
#                         edges.append((mnemonic_node_idx, mem_node_idx, self.edge_types['output']))
#
#             # Process source operands (inputs)
#             for src_idx, src_operand in enumerate(operands[1:], 1):
#                 # Check if it's a register
#                 if src_operand in self.token_to_idx and src_operand.startswith(
#                         ('x', 'a', 's', 't', 'ra', 'sp', 'gp', 'tp', 'fp', 'zero')):
#                     # Use existing register node or create a new one
#                     if src_operand in value_nodes:
#                         src_node_idx = value_nodes[src_operand]
#                     else:
#                         src_node_idx = node_idx
#                         nodes.append((self.node_types['register'], self.token_to_idx.get(src_operand, 0)))
#                         value_nodes[src_operand] = src_node_idx
#                         node_idx += 1
#
#                     # Add input edge from register to instruction
#                     edges.append((src_node_idx, mnemonic_node_idx, self.edge_types['input']))
#
#                 # Check if it's an immediate value
#                 elif src_operand.lstrip('-').isdigit() or (
#                         src_operand.startswith('0x') and all(c in '0123456789abcdefABCDEF' for c in src_operand[2:])):
#                     # Create immediate node
#                     imm_node_idx = node_idx
#                     nodes.append((self.node_types['immediate'], self.token_to_idx['<IMM>']))
#                     node_idx += 1
#
#                     # Add input edge from immediate to instruction
#                     edges.append((imm_node_idx, mnemonic_node_idx, self.edge_types['input']))
#
#                 # Check if it's a memory load
#                 elif '(' in src_operand and ')' in src_operand:
#                     # Memory load like "lw x1, 8(x2)"
#                     # Extract offset and base register
#                     match = re.match(r'(\d+)\(([^\)]+)\)', src_operand)
#                     if match:
#                         offset, base_reg = match.groups()
#
#                         # Create address node
#                         addr_node_idx = node_idx
#                         nodes.append((self.node_types['address'], self.token_to_idx['<ADDR>']))
#                         node_idx += 1
#
#                         # Create memory node
#                         mem_node_idx = node_idx
#                         nodes.append((self.node_types['memory'], self.token_to_idx['<MEM>']))
#                         node_idx += 1
#
#                         # Add base register to address edge
#                         if base_reg in value_nodes:
#                             edges.append((value_nodes[base_reg], addr_node_idx, self.edge_types['address_base']))
#
#                         # Add immediate offset to address edge
#                         imm_node_idx = node_idx
#                         nodes.append((self.node_types['immediate'], self.token_to_idx['<IMM>']))
#                         node_idx += 1
#                         edges.append((imm_node_idx, addr_node_idx, self.edge_types['address_offset']))
#
#                         # Add input edge from memory to instruction
#                         edges.append((mem_node_idx, mnemonic_node_idx, self.edge_types['input']))
#
#         # Create tensors for PyTorch Geometric
#         x = torch.zeros((len(nodes), 2), dtype=torch.long)
#         edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
#         edge_attr = torch.zeros((len(edges), 1), dtype=torch.long)
#
#         # Fill node features: type and token
#         for i, (node_type, token_idx) in enumerate(nodes):
#             x[i, 0] = node_type
#             x[i, 1] = token_idx
#
#         # Fill edge features
#         for i, (src, dst, edge_type) in enumerate(edges):
#             edge_index[0, i] = src
#             edge_index[1, i] = dst
#             edge_attr[i, 0] = edge_type
#
#         # Create mask for instruction nodes
#         instruction_mask = torch.zeros(len(nodes), dtype=torch.bool)
#         for idx in instruction_nodes:
#             instruction_mask[idx] = True
#
#         # Create PyTorch Geometric Data object
#         data = torch_geometric.data.Data(
#             x=x,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             instruction_mask=instruction_mask,
#             num_nodes=len(nodes)
#         )
#
#         return data
#
#
# class GraphNeuralNetwork(nn.Module):
#     """
#     Graph Neural Network implementation following the GRANITE paper.
#     Uses message passing to learn node, edge, and global embeddings.
#     """
#
#     def __init__(
#             self,
#             node_embedding_dim: int = 256,
#             edge_embedding_dim: int = 256,
#             global_embedding_dim: int = 256,
#             hidden_dim: int = 256,
#             num_message_passing_steps: int = 8,
#             dropout: float = 0.1,
#             use_layer_norm: bool = True,
#     ):
#         """
#         Initialize the Graph Neural Network.
#
#         Args:
#             node_embedding_dim: Dimension of node embeddings
#             edge_embedding_dim: Dimension of edge embeddings
#             global_embedding_dim: Dimension of global embeddings
#             hidden_dim: Dimension of hidden layers
#             num_message_passing_steps: Number of message passing iterations
#             dropout: Dropout rate
#             use_layer_norm: Whether to use layer normalization
#         """
#         super(GraphNeuralNetwork, self).__init__()
#
#         self.node_embedding_dim = node_embedding_dim
#         self.edge_embedding_dim = edge_embedding_dim
#         self.global_embedding_dim = global_embedding_dim
#         self.hidden_dim = hidden_dim
#         self.num_message_passing_steps = num_message_passing_steps
#         self.use_layer_norm = use_layer_norm
#
#         # Node type and token embeddings
#         self.node_type_embedding = nn.Embedding(10, node_embedding_dim // 2)  # Assuming at most 10 node types
#         self.node_token_embedding = nn.Embedding(1000, node_embedding_dim // 2)  # Assuming at most 1000 tokens
#
#         # Edge type embeddings
#         self.edge_type_embedding = nn.Embedding(10, edge_embedding_dim)  # Assuming at most 10 edge types
#
#         # Global feature initialization
#         self.global_init = nn.Linear(1, global_embedding_dim)
#
#         # Message passing layers
#         self.message_passing_layers = nn.ModuleList([
#             MessagePassingLayer(
#                 node_embedding_dim=node_embedding_dim,
#                 edge_embedding_dim=edge_embedding_dim,
#                 global_embedding_dim=global_embedding_dim,
#                 hidden_dim=hidden_dim,
#                 dropout=dropout,
#                 use_layer_norm=use_layer_norm
#             ) for _ in range(num_message_passing_steps)
#         ])
#
#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
#                 edge_attr: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[
#         torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Forward pass of the GNN.
#
#         Args:
#             x: Node features [num_nodes, 2] (type, token)
#             edge_index: Graph connectivity [2, num_edges]
#             edge_attr: Edge attributes [num_edges, 1]
#             batch: Batch assignment for nodes [num_nodes]
#
#         Returns:
#             node_embeddings: Updated node embeddings
#             edge_embeddings: Updated edge embeddings
#             global_embedding: Updated global embedding
#         """
#         # Initialize node embeddings
#         node_type_emb = self.node_type_embedding(x[:, 0])
#         node_token_emb = self.node_token_embedding(x[:, 1])
#         node_embeddings = torch.cat([node_type_emb, node_token_emb], dim=1)
#
#         # Initialize edge embeddings
#         edge_embeddings = self.edge_type_embedding(edge_attr.squeeze(-1))
#
#         # Initialize global features
#         if batch is None:
#             # Single graph case
#             global_embedding = self.global_init(torch.ones(1, 1, device=x.device))
#         else:
#             # Batched graphs case
#             num_graphs = batch.max().item() + 1
#             global_embedding = self.global_init(torch.ones(num_graphs, 1, device=x.device))
#
#         # Apply message passing
#         for i in range(self.num_message_passing_steps):
#             node_embeddings, edge_embeddings, global_embedding = self.message_passing_layers[i](
#                 node_embeddings, edge_embeddings, global_embedding, edge_index, batch
#             )
#
#         return node_embeddings, edge_embeddings, global_embedding
#
#
# class MessagePassingLayer(nn.Module):
#     """
#     A single message passing layer for the GNN, implementing the 'full GN block'
#     architecture from the paper 'Relational inductive biases, deep learning, and graph networks'.
#     """
#
#     def __init__(
#             self,
#             node_embedding_dim: int,
#             edge_embedding_dim: int,
#             global_embedding_dim: int,
#             hidden_dim: int,
#             dropout: float = 0.1,
#             use_layer_norm: bool = True,
#     ):
#         """
#         Initialize the message passing layer.
#
#         Args:
#             node_embedding_dim: Dimension of node embeddings
#             edge_embedding_dim: Dimension of edge embeddings
#             global_embedding_dim: Dimension of global embeddings
#             hidden_dim: Dimension of hidden layers
#             dropout: Dropout rate
#             use_layer_norm: Whether to use layer normalization
#         """
#         super(MessagePassingLayer, self).__init__()
#
#         self.node_embedding_dim = node_embedding_dim
#         self.edge_embedding_dim = edge_embedding_dim
#         self.global_embedding_dim = global_embedding_dim
#         self.hidden_dim = hidden_dim
#         self.use_layer_norm = use_layer_norm
#
#         # Edge update network
#         self.edge_update = nn.Sequential(
#             nn.Linear(edge_embedding_dim + 2 * node_embedding_dim + global_embedding_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, edge_embedding_dim)
#         )
#
#         # Node update network
#         self.node_update = nn.Sequential(
#             nn.Linear(node_embedding_dim + hidden_dim + global_embedding_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, node_embedding_dim)
#         )
#
#         # Global update network
#         self.global_update = nn.Sequential(
#             nn.Linear(global_embedding_dim + hidden_dim + hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, global_embedding_dim)
#         )
#
#         # Layer normalization
#         if use_layer_norm:
#             self.edge_layer_norm = nn.LayerNorm(edge_embedding_dim)
#             self.node_layer_norm = nn.LayerNorm(node_embedding_dim)
#             self.global_layer_norm = nn.LayerNorm(global_embedding_dim)
#
#         # Projection matrices for message aggregation
#         self.edge_to_message = nn.Linear(edge_embedding_dim, hidden_dim)
#         self.node_to_global = nn.Linear(node_embedding_dim, hidden_dim)
#         self.edge_to_global = nn.Linear(edge_embedding_dim, hidden_dim)
#
#     def forward(self, node_embeddings: torch.Tensor, edge_embeddings: torch.Tensor,
#                 global_embedding: torch.Tensor, edge_index: torch.Tensor,
#                 batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Forward pass of a single message passing layer.
#
#         Args:
#             node_embeddings: Current node embeddings [num_nodes, node_embedding_dim]
#             edge_embeddings: Current edge embeddings [num_edges, edge_embedding_dim]
#             global_embedding: Current global embedding [num_graphs, global_embedding_dim]
#             edge_index: Graph connectivity [2, num_edges]
#             batch: Batch assignment for nodes [num_nodes]
#
#         Returns:
#             updated_node_embeddings: Updated node embeddings
#             updated_edge_embeddings: Updated edge embeddings
#             updated_global_embedding: Updated global embedding
#         """
#         num_nodes = node_embeddings.size(0)
#         num_edges = edge_index.size(1)
#
#         # Default batch for a single graph
#         if batch is None:
#             batch = torch.zeros(num_nodes, dtype=torch.long, device=node_embeddings.device)
#
#         # Edge update
#         src_nodes = edge_index[0]
#         dst_nodes = edge_index[1]
#
#         src_embeddings = node_embeddings[src_nodes]
#         dst_embeddings = node_embeddings[dst_nodes]
#
#         # For each edge, get the global embedding of its graph
#         edge_global_embeddings = global_embedding[batch[src_nodes]]
#
#         # Concatenate source, destination, edge, and global features
#         edge_inputs = torch.cat([
#             src_embeddings,
#             dst_embeddings,
#             edge_embeddings,
#             edge_global_embeddings
#         ], dim=1)
#
#         # Apply edge update network
#         edge_updates = self.edge_update(edge_inputs)
#
#         # Apply residual connection and layer normalization
#         updated_edge_embeddings = edge_embeddings + edge_updates
#         if self.use_layer_norm:
#             updated_edge_embeddings = self.edge_layer_norm(updated_edge_embeddings)
#
#         # Aggregate edge messages for each node
#         # Project edge embeddings to messages
#         edge_messages = self.edge_to_message(updated_edge_embeddings)
#
#         # Initialize node messages
#         node_messages = torch.zeros(num_nodes, self.hidden_dim, device=node_embeddings.device)
#
#         # Aggregate messages from incoming edges (to the destination nodes)
#         for i in range(num_edges):
#             node_messages[dst_nodes[i]] += edge_messages[i]
#
#         # Node update
#         # For each node, get the global embedding of its graph
#         node_global_embeddings = global_embedding[batch]
#
#         # Concatenate node features, aggregated messages, and global features
#         node_inputs = torch.cat([
#             node_embeddings,
#             node_messages,
#             node_global_embeddings
#         ], dim=1)
#
#         # Apply node update network
#         node_updates = self.node_update(node_inputs)
#
#         # Apply residual connection and layer normalization
#         updated_node_embeddings = node_embeddings + node_updates
#         if self.use_layer_norm:
#             updated_node_embeddings = self.node_layer_norm(updated_node_embeddings)
#
#         # Global update
#         # Aggregate node and edge features for global update
#         num_graphs = global_embedding.size(0)
#
#         # Project node embeddings for global aggregation
#         node_features_for_global = self.node_to_global(updated_node_embeddings)
#
#         # Project edge embeddings for global aggregation
#         edge_features_for_global = self.edge_to_global(updated_edge_embeddings)
#
#         # Aggregate node embeddings per graph
#         node_aggregated = torch.zeros(num_graphs, self.hidden_dim, device=node_embeddings.device)
#         for i in range(num_nodes):
#             node_aggregated[batch[i]] += node_features_for_global[i]
#
#         # Aggregate edge embeddings per graph
#         edge_aggregated = torch.zeros(num_graphs, self.hidden_dim, device=edge_embeddings.device)
#         for i in range(num_edges):
#             edge_aggregated[batch[src_nodes[i]]] += edge_features_for_global[i]
#
#         # Concatenate global features and aggregated node and edge features
#         global_inputs = torch.cat([
#             global_embedding,
#             node_aggregated,
#             edge_aggregated
#         ], dim=1)
#
#         # Apply global update network
#         global_updates = self.global_update(global_inputs)
#
#         # Apply residual connection and layer normalization
#         updated_global_embedding = global_embedding + global_updates
#         if self.use_layer_norm:
#             updated_global_embedding = self.global_layer_norm(updated_global_embedding)
#
#         return updated_node_embeddings, updated_edge_embeddings, updated_global_embedding
#
#
# class ThroughputDecoder(nn.Module):
#     """
#     Decoder network to predict throughput from instruction embeddings.
#     Following the GRANITE model, it computes the contribution of each
#     instruction to the overall throughput and then sums them.
#     """
#
#     def __init__(
#             self,
#             node_embedding_dim: int = 256,
#             hidden_dim: int = 256,
#             dropout: float = 0.1,
#             use_layer_norm: bool = True,
#     ):
#         """
#         Initialize the throughput decoder.
#
#         Args:
#             node_embedding_dim: Dimension of node embeddings
#             hidden_dim: Dimension of hidden layers
#             dropout: Dropout rate
#             use_layer_norm: Whether to use layer normalization
#         """
#         super(ThroughputDecoder, self).__init__()
#
#         # Two-layer feed-forward ReLU network with residual connections
#         self.decoder = nn.Sequential(
#             nn.Linear(node_embedding_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )
#
#         self.use_layer_norm = use_layer_norm
#         if use_layer_norm:
#             self.layer_norm = nn.LayerNorm(node_embedding_dim)
#
#     def forward(self, instruction_embeddings: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the decoder.
#
#         Args:
#             instruction_embeddings: Embeddings of instruction nodes [num_instructions, node_embedding_dim]
#
#         Returns:
#             throughput: Predicted throughput of the basic block
#         """
#         if self.use_layer_norm:
#             instruction_embeddings = self.layer_norm(instruction_embeddings)
#
#         # Apply decoder to each instruction embedding to get per-instruction contribution
#         instruction_contributions = self.decoder(instruction_embeddings)
#
#         # Sum all instruction contributions to get the final throughput
#         throughput = torch.sum(instruction_contributions)
#
#         return throughput
#

#############################################
# GNN模型定义
#############################################

class GNNModel(nn.Module):
    """
    用于RISC-V吞吐量预测的图神经网络模型

    这个模型结合了图编码器和GNN模型来预测RISC-V基本块的吞吐量
    """

    def __init__(self, config):
        """
        初始化GNN模型

        Args:
            config: 包含模型超参数的配置对象
        """
        super(GNNModel, self).__init__()

        # 创建RISC-V GRANITE模型
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
        """返回模型的可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch):
        """
        模型的前向传播

        Args:
            batch: 包含图数据的批次

        Returns:
            预测的吞吐量值
        """
        # 批次中已经包含了预处理好的图，直接使用
        throughput = self.model(batch)
        print(batch)
        return throughput


class RISCVGraniteModel(nn.Module):
    """
    基于图神经网络的RISC-V基本块吞吐量估计模型
    受GRANITE论文启发
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
        """
        初始化RISC-V GRANITE模型

        Args:
            node_embedding_dim: 节点嵌入维度
            edge_embedding_dim: 边嵌入维度
            global_embedding_dim: 全局嵌入维度
            hidden_dim: 隐藏层维度
            num_message_passing_steps: 消息传递迭代次数
            dropout: Dropout率
            use_layer_norm: 是否使用层归一化
        """
        super(RISCVGraniteModel, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.global_embedding_dim = global_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.use_layer_norm = use_layer_norm

        # 初始化GNN
        self.gnn = GraphNeuralNetwork(
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            global_embedding_dim=global_embedding_dim,
            hidden_dim=hidden_dim,
            num_message_passing_steps=num_message_passing_steps,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        # 初始化解码器
        self.decoder = ThroughputDecoder(
            node_embedding_dim=node_embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    # def forward(self, basic_block_graph: torch_geometric.data.Data) -> torch.Tensor:
    #     """
    #     模型的前向传播
    #
    #     Args:
    #         basic_block_graph: RISC-V基本块的图表示
    #
    #     Returns:
    #         throughput: 基本块的估计吞吐量
    #     """
    #     # 应用GNN获取节点嵌入
    #     node_embeddings, _, _ = self.gnn(
    #         basic_block_graph.x,
    #         basic_block_graph.edge_index,
    #         basic_block_graph.edge_attr,
    #         basic_block_graph.batch if hasattr(basic_block_graph, 'batch') else None
    #     )
    #
    #     # 使用指令掩码获取指令节点
    #     instruction_mask = basic_block_graph.instruction_mask
    #     instruction_embeddings = node_embeddings[instruction_mask]
    #
    #     # 应用解码器获取吞吐量估计
    #     throughput = self.decoder(instruction_embeddings)
    #
    #     return throughput
    def forward(self, basic_block_graph):
        """
        模型的前向传播

        Args:
            basic_block_graph: RISC-V基本块的图表示（或批次）

        Returns:
            throughputs: 每个样本的估计吞吐量 [batch_size]
        """
        # 应用GNN获取节点嵌入
        node_embeddings, _, _ = self.gnn(
            basic_block_graph.x,
            basic_block_graph.edge_index,
            basic_block_graph.edge_attr,
            basic_block_graph.batch if hasattr(basic_block_graph, 'batch') else None
        )

        # 获取指令节点
        instruction_mask = basic_block_graph.instruction_mask
        instruction_embeddings = node_embeddings[instruction_mask]

        # 获取指令节点的批次分配
        if hasattr(basic_block_graph, 'batch'):
            # 仅保留指令节点的批次分配
            instruction_batch = basic_block_graph.batch[instruction_mask]
        else:
            instruction_batch = None

        # 应用解码器获取每个样本的吞吐量估计
        throughputs = self.decoder(instruction_embeddings, instruction_batch)

        return throughputs


class GraphNeuralNetwork(nn.Module):
    """
    遵循GRANITE论文的图神经网络实现
    使用消息传递学习节点、边和全局嵌入
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
        """
        初始化图神经网络

        Args:
            node_embedding_dim: 节点嵌入维度
            edge_embedding_dim: 边嵌入维度
            global_embedding_dim: 全局嵌入维度
            hidden_dim: 隐藏层维度
            num_message_passing_steps: 消息传递迭代次数
            dropout: Dropout率
            use_layer_norm: 是否使用层归一化
        """
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
        """
        GNN的前向传播

        Args:
            x: 节点特征 [num_nodes, 2] (类型, token)
            edge_index: 图连接关系 [2, num_edges]
            edge_attr: 边属性 [num_edges, 1]
            batch: 节点的批次分配 [num_nodes]

        Returns:
            node_embeddings: 更新的节点嵌入
            edge_embeddings: 更新的边嵌入
            global_embedding: 更新的全局嵌入
        """
        # 初始化节点嵌入
        node_type_emb = self.node_type_embedding(x[:, 0])
        node_token_emb = self.node_token_embedding(x[:, 1])
        node_embeddings = torch.cat([node_type_emb, node_token_emb], dim=1)

        # 初始化边嵌入
        edge_embeddings = self.edge_type_embedding(edge_attr.squeeze(-1))

        # 初始化全局特征
        if batch is None:
            # 单个图的情况
            global_embedding = self.global_init(torch.ones(1, 1, device=x.device))
        else:
            # 批处理图的情况
            num_graphs = batch.max().item() + 1
            global_embedding = self.global_init(torch.ones(num_graphs, 1, device=x.device))

        # 应用消息传递
        for i in range(self.num_message_passing_steps):
            node_embeddings, edge_embeddings, global_embedding = self.message_passing_layers[i](
                node_embeddings, edge_embeddings, global_embedding, edge_index, batch
            )

        return node_embeddings, edge_embeddings, global_embedding


class MessagePassingLayer(nn.Module):
    """
    GNN的单个消息传递层，实现了来自论文"Relational inductive biases, deep learning, and graph networks"
    的'full GN block'架构
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
        """
        初始化消息传递层

        Args:
            node_embedding_dim: 节点嵌入维度
            edge_embedding_dim: 边嵌入维度
            global_embedding_dim: 全局嵌入维度
            hidden_dim: 隐藏层维度
            dropout: Dropout率
            use_layer_norm: 是否使用层归一化
        """
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
        """
        单个消息传递层的前向传播

        Args:
            node_embeddings: 当前节点嵌入 [num_nodes, node_embedding_dim]
            edge_embeddings: 当前边嵌入 [num_edges, edge_embedding_dim]
            global_embedding: 当前全局嵌入 [num_graphs, global_embedding_dim]
            edge_index: 图连接关系 [2, num_edges]
            batch: 节点的批次分配 [num_nodes]

        Returns:
            updated_node_embeddings: 更新的节点嵌入
            updated_edge_embeddings: 更新的边嵌入
            updated_global_embedding: 更新的全局嵌入
        """
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


# class ThroughputDecoder(nn.Module):
#     """
#     从指令嵌入预测吞吐量的解码器网络
#     遵循GRANITE模型，计算每条指令对整体吞吐量的贡献，然后求和
#     """
#
#     def __init__(
#             self,
#             node_embedding_dim: int = 256,
#             hidden_dim: int = 256,
#             dropout: float = 0.1,
#             use_layer_norm: bool = True,
#     ):
#         """
#         初始化吞吐量解码器
#
#         Args:
#             node_embedding_dim: 节点嵌入维度
#             hidden_dim: 隐藏层维度
#             dropout: Dropout率
#             use_layer_norm: 是否使用层归一化
#         """
#         super(ThroughputDecoder, self).__init__()
#
#         # 两层前馈ReLU网络，带残差连接
#         self.decoder = nn.Sequential(
#             nn.Linear(node_embedding_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )
#
#         self.use_layer_norm = use_layer_norm
#         if use_layer_norm:
#             self.layer_norm = nn.LayerNorm(node_embedding_dim)
#
#     def forward(self, instruction_embeddings: torch.Tensor) -> torch.Tensor:
#         """
#         解码器的前向传播
#
#         Args:
#             instruction_embeddings: 指令节点嵌入 [num_instructions, node_embedding_dim]
#
#         Returns:
#             throughput: 基本块的预测吞吐量
#         """
#         if self.use_layer_norm:
#             instruction_embeddings = self.layer_norm(instruction_embeddings)
#
#         # 对每个指令嵌入应用解码器，获取每条指令的贡献
#         instruction_contributions = self.decoder(instruction_embeddings)
#
#         # 求和所有指令贡献，得到最终吞吐量
#         throughput = torch.sum(instruction_contributions)
#
#         return throughput

class ThroughputDecoder(nn.Module):
    """
    从指令嵌入预测吞吐量的解码器网络
    """

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