import torch
import torch_geometric
import re
from typing import Dict, List, Tuple, Set, Optional

class RISCVGraphEncoder:
    """
    Encodes RISC-V basic blocks into graph representations.
    """
    def __init__(self):
        # Define node types
        self.node_types = {
            'mnemonic': 0,     # Instruction mnemonic (e.g., 'addi')
            'register': 1,     # Register (e.g., 'x1')
            'immediate': 2,    # Immediate value
            'memory': 3,       # Memory value
            'address': 4,      # Address computation
            'prefix': 5,       # Instruction prefix (rarely used in RISC-V but included for completeness)
        }
        
        # Define edge types
        self.edge_types = {
            'structural': 0,   # From one instruction to the next
            'input': 1,        # From value node to instruction node
            'output': 2,       # From instruction node to value node
            'address_base': 3, # From register to address
            'address_offset': 4, # From immediate to address
        }
        
        # Define token to index mappings
        self.token_to_idx = {}
        self.edge_type_to_idx = {}
        
        # Initialize with special tokens
        self.token_to_idx['<UNK>'] = 0
        
        # RISC-V instruction mnemonics (just a subset for illustration)
        # risc_v_instructions = [
        #     'add', 'addi', 'sub', 'lui', 'auipc', 'jal', 'jalr', 'beq', 'bne', 'blt', 'bge',
        #     'bltu', 'bgeu', 'lb', 'lh', 'lw', 'lbu', 'lhu', 'sb', 'sh', 'sw', 'sll', 'slli',
        #     'srl', 'srli', 'sra', 'srai', 'and', 'andi', 'or', 'ori', 'xor', 'xori',
        #     'slti', 'sltiu', 'slt', 'sltu', 'mul', 'mulh', 'div', 'rem'
        # ]
        risc_v_instructions = ['amoadd.w', 'amoand.w', 'amomax.w', 'amomaxu.w', 'amomin.w', 'amominu.w', 'amoor.w', 'amoswap.w',
            'amoxor.w', 'lr.w', 'sc.w', 'mul', 'mulh', 'mulhsu', 'mulhu', 'div', 'divu', 'rem', 'remu', 'add',
            'addi', 'sub', 'lui', 'auipc', 'sll', 'slli', 'srl', 'srli', 'sra', 'srai', 'slt', 'slti', 'sltiu',
            'sltu', 'and', 'andi', 'or', 'ori', 'xor', 'xori', 'beq', 'bge', 'bgeu', 'blt', 'bltu', 'bne', 'lb',
            'lbu', 'lh', 'lhu', 'lw', 'sb', 'sh', 'sw', 'jal', 'jalr', 'ebreak', 'ecall', 'fadd.d', 'fadd.s',
            'fclass.d', 'fclass.s', 'fcvt.d.s', 'fcvt.d.w', 'fcvt.d.wu', 'fcvt.s.d', 'fcvt.s.w', 'fcvt.s.wu',
            'fcvt.w.d', 'fcvt.w.s', 'fcvt.wu.d', 'fcvt.wu.s', 'fdiv.d', 'fdiv.s', 'fence', 'fence.i', 'feq.d',
            'feq.s', 'fld', 'fle.d', 'fle.s', 'flt.d', 'flt.s', 'fsw', 'flw', 'fmadd.d', 'fmadd.s', 'fmax.d',
            'fmax.s', 'fmin.d', 'fmin.s', 'fmsub.d', 'fmsub.s', 'fmul.d', 'fmul.s', 'fmv.w.x', 'fmv.x.w',
            'fnmadd.d', 'fnmadd.s', 'fnmsub.d', 'fnmsub.s', 'fsd', 'fsgnj.d', 'fsgnj.s', 'fsgnjn.d', 'fsgnjn.s',
            'fsgnjx.d', 'fsgnjx.s', 'fsqrt.d', 'fsqrt.s', 'fsub.d', 'fsub.s', 'csrrc', 'csrrci', 'csrrs',
            'csrrsi', 'csrrw', 'csrrwi', 'amoadd.d', 'amoand.d', 'amomax.d', 'amomaxu.d', 'amomin.d',
            'amominu.d', 'amoor.d', 'amoswap.d', 'amoxor.d', 'lr.d', 'sc.d', 'mulw', 'divw', 'divuw', 'remw',
            'remuw', 'addiw', 'addw', 'subw', 'srliw', 'srlw', 'slliw', 'sllw', 'sraiw', 'sraw', 'lwu', 'ld',
            'sd', 'fmv.d.x', 'fmv.x.d', 'fcvt.s.l', 'fcvt.s.lu', 'fcvt.lu.d', 'fcvt.lu.s', 'fcvt.l.d',
            'fcvt.l.s', 'fcvt.d.l', 'fcvt.d.lu']

        # Add RISC-V instructions to token mapping
        for i, instr in enumerate(risc_v_instructions):
            self.token_to_idx[instr] = i + 1  # +1 because 0 is <UNK>
            
        # RISC-V registers (x0-x31, plus aliases)
        registers = ['x' + str(i) for i in range(32)]
        register_aliases = {
            'zero': 'x0', 'ra': 'x1', 'sp': 'x2', 'gp': 'x3', 'tp': 'x4',
            't0': 'x5', 't1': 'x6', 't2': 'x7', 's0': 'x8', 'fp': 'x8',
            's1': 'x9', 'a0': 'x10', 'a1': 'x11', 'a2': 'x12', 'a3': 'x13',
            'a4': 'x14', 'a5': 'x15', 'a6': 'x16', 'a7': 'x17',
            's2': 'x18', 's3': 'x19', 's4': 'x20', 's5': 'x21', 's6': 'x22',
            's7': 'x23', 's8': 'x24', 's9': 'x25', 's10': 'x26', 's11': 'x27',
            't3': 'x28', 't4': 'x29', 't5': 'x30', 't6': 'x31'
        }
        
        # Add RISC-V registers to token mapping
        next_idx = len(self.token_to_idx)
        for i, reg in enumerate(registers):
            self.token_to_idx[reg] = next_idx + i
            
        # Add register aliases
        for alias, reg in register_aliases.items():
            self.token_to_idx[alias] = self.token_to_idx[reg]
            
        # Special tokens for immediate values, memory, and address computation
        self.token_to_idx['<IMM>'] = len(self.token_to_idx)
        self.token_to_idx['<MEM>'] = len(self.token_to_idx)
        self.token_to_idx['<ADDR>'] = len(self.token_to_idx)
        
        # Map edge types to indices
        for i, edge_type in enumerate(self.edge_types.keys()):
            self.edge_type_to_idx[edge_type] = i
    
    def parse_instruction(self, instruction: str) -> Dict:
        """
        Parse a RISC-V instruction into its components.
        
        Args:
            instruction: A RISC-V assembly instruction string
            
        Returns:
            A dictionary with instruction components
        """
        instruction = instruction.strip().lower()
        
        # Simple regex to extract mnemonic and operands
        match = re.match(r'([a-z0-9\.]+)\s*(.*)', instruction)
        if not match:
            return {'mnemonic': '<UNK>', 'operands': []}
        
        mnemonic, operands_str = match.groups()
        
        # Split operands by commas, handling potential spaces
        operands = [op.strip() for op in operands_str.split(',')] if operands_str else []
        
        return {
            'mnemonic': mnemonic,
            'operands': operands
        }
    
    def build_graph(self, basic_block: List[str]) -> torch_geometric.data.Data:
        """
        Build a graph representation of a RISC-V basic block.
        
        Args:
            basic_block: List of RISC-V assembly instructions
            
        Returns:
            A PyTorch Geometric Data object representing the graph
        """
        nodes = []  # (type, token)
        edges = []  # (src, dst, type)
        
        # Maps to track nodes for values (registers, memory, etc.)
        value_nodes = {}  # Maps value name to node index
        instruction_nodes = []  # List of instruction node indices
        
        node_idx = 0
        
        for instr_idx, instruction in enumerate(basic_block):
            parsed = self.parse_instruction(instruction)
            mnemonic = parsed['mnemonic']
            operands = parsed['operands']
            
            # Add instruction mnemonic node
            mnemonic_node_idx = node_idx
            nodes.append((self.node_types['mnemonic'], self.token_to_idx.get(mnemonic, 0)))
            instruction_nodes.append(mnemonic_node_idx)
            node_idx += 1
            
            # Add structural dependency edge to previous instruction
            if instr_idx > 0:
                edges.append((instruction_nodes[instr_idx-1], mnemonic_node_idx, self.edge_types['structural']))
            
            # Process destination operands (outputs)
            if len(operands) > 0:
                dest_operand = operands[0]
                
                # Check if it's a register
                if dest_operand in self.token_to_idx and dest_operand.startswith(('x', 'a', 's', 't', 'ra', 'sp', 'gp', 'tp', 'fp', 'zero')):
                    # Create a new register node for the destination
                    dest_node_idx = node_idx
                    nodes.append((self.node_types['register'], self.token_to_idx.get(dest_operand, 0)))
                    node_idx += 1
                    
                    # Add output edge from instruction to register
                    edges.append((mnemonic_node_idx, dest_node_idx, self.edge_types['output']))
                    
                    # Update value_nodes to point to this new node
                    value_nodes[dest_operand] = dest_node_idx
                
                # Check if it's a memory store
                elif '(' in dest_operand and ')' in dest_operand:
                    # Memory store like "sw x1, 8(x2)"
                    # Extract offset and base register
                    match = re.match(r'(\d+)\(([^\)]+)\)', dest_operand)
                    if match:
                        offset, base_reg = match.groups()
                        
                        # Create address node
                        addr_node_idx = node_idx
                        nodes.append((self.node_types['address'], self.token_to_idx['<ADDR>']))
                        node_idx += 1
                        
                        # Create memory node
                        mem_node_idx = node_idx
                        nodes.append((self.node_types['memory'], self.token_to_idx['<MEM>']))
                        node_idx += 1
                        
                        # Add base register to address edge
                        if base_reg in value_nodes:
                            edges.append((value_nodes[base_reg], addr_node_idx, self.edge_types['address_base']))
                        
                        # Add immediate offset to address edge
                        imm_node_idx = node_idx
                        nodes.append((self.node_types['immediate'], self.token_to_idx['<IMM>']))
                        node_idx += 1
                        edges.append((imm_node_idx, addr_node_idx, self.edge_types['address_offset']))
                        
                        # Add output edge from instruction to memory
                        edges.append((mnemonic_node_idx, mem_node_idx, self.edge_types['output']))
            
            # Process source operands (inputs)
            for src_idx, src_operand in enumerate(operands[1:], 1):
                # Check if it's a register
                if src_operand in self.token_to_idx and src_operand.startswith(('x', 'a', 's', 't', 'ra', 'sp', 'gp', 'tp', 'fp', 'zero')):
                    # Use existing register node or create a new one
                    if src_operand in value_nodes:
                        src_node_idx = value_nodes[src_operand]
                    else:
                        src_node_idx = node_idx
                        nodes.append((self.node_types['register'], self.token_to_idx.get(src_operand, 0)))
                        value_nodes[src_operand] = src_node_idx
                        node_idx += 1
                    
                    # Add input edge from register to instruction
                    edges.append((src_node_idx, mnemonic_node_idx, self.edge_types['input']))
                
                # Check if it's an immediate value
                elif src_operand.lstrip('-').isdigit() or (src_operand.startswith('0x') and all(c in '0123456789abcdefABCDEF' for c in src_operand[2:])):
                    # Create immediate node
                    imm_node_idx = node_idx
                    nodes.append((self.node_types['immediate'], self.token_to_idx['<IMM>']))
                    node_idx += 1
                    
                    # Add input edge from immediate to instruction
                    edges.append((imm_node_idx, mnemonic_node_idx, self.edge_types['input']))
                
                # Check if it's a memory load
                elif '(' in src_operand and ')' in src_operand:
                    # Memory load like "lw x1, 8(x2)"
                    # Extract offset and base register
                    match = re.match(r'(\d+)\(([^\)]+)\)', src_operand)
                    if match:
                        offset, base_reg = match.groups()
                        
                        # Create address node
                        addr_node_idx = node_idx
                        nodes.append((self.node_types['address'], self.token_to_idx['<ADDR>']))
                        node_idx += 1
                        
                        # Create memory node
                        mem_node_idx = node_idx
                        nodes.append((self.node_types['memory'], self.token_to_idx['<MEM>']))
                        node_idx += 1
                        
                        # Add base register to address edge
                        if base_reg in value_nodes:
                            edges.append((value_nodes[base_reg], addr_node_idx, self.edge_types['address_base']))
                        
                        # Add immediate offset to address edge
                        imm_node_idx = node_idx
                        nodes.append((self.node_types['immediate'], self.token_to_idx['<IMM>']))
                        node_idx += 1
                        edges.append((imm_node_idx, addr_node_idx, self.edge_types['address_offset']))
                        
                        # Add input edge from memory to instruction
                        edges.append((mem_node_idx, mnemonic_node_idx, self.edge_types['input']))
        
        # Create tensors for PyTorch Geometric
        x = torch.zeros((len(nodes), 2), dtype=torch.long)
        edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
        edge_attr = torch.zeros((len(edges), 1), dtype=torch.long)
        
        # Fill node features: type and token
        for i, (node_type, token_idx) in enumerate(nodes):
            x[i, 0] = node_type
            x[i, 1] = token_idx
        
        # Fill edge features
        for i, (src, dst, edge_type) in enumerate(edges):
            edge_index[0, i] = src
            edge_index[1, i] = dst
            edge_attr[i, 0] = edge_type
        
        # Create mask for instruction nodes
        instruction_mask = torch.zeros(len(nodes), dtype=torch.bool)
        for idx in instruction_nodes:
            instruction_mask[idx] = True
        
        # Create PyTorch Geometric Data object
        data = torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            instruction_mask=instruction_mask,
            num_nodes=len(nodes)
        )
        
        return data
