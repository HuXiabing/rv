import os
import json
import time
import torch
import torch_geometric
from torch.utils.data import Dataset
from typing import Dict, List
import re
from tqdm import tqdm

class RISCVGraphEncoder:

    def __init__(self, predefined_token_map="data/vocab.dump"):

        self.node_types = {
            'mnemonic': 0,  # 指令助记符 (例如, 'addi')
            'register': 1,  # 寄存器 (例如, 'x1')
            'immediate': 2,  # 立即数
            'memory': 3,  # 内存值
            'address': 4,  # 地址计算
            'prefix': 5,  # 指令前缀
        }

        self.edge_types = {
            'structural': 0,  # 从一条指令到下一条
            'input': 1,  # 从值节点到指令节点
            'output': 2,  # 从指令节点到值节点
            'address_base': 3,  # 从寄存器到地址
            'address_offset': 4,  # 从立即数到地址
        }

        if predefined_token_map is not None:
            self.token_to_idx = torch.load(predefined_token_map)
        else:
            self.token_to_idx = {'<PAD>': 0, '<BLOCK_START>': 1, '<BLOCK_END>': 2, '<ADDRESS>': 3, '<E>': 4, '<D>': 5, '<S>': 6, '<CONST>': 7, '<CSR>': 8,
        'zero': 9, 'ra': 10, 'sp': 11, 'gp': 12, 'tp': 13, 't0': 14, 't1': 15, 't2': 16, 's0': 17, 's1': 18, 'a0': 19, 'a1': 20,
        'a2': 21, 'a3': 22, 'a4': 23, 'a5': 24, 'a6': 25, 'a7': 26, 's2': 27, 's3': 28, 's4': 29, 's5': 30, 's6': 31, 's7': 32,
        's8': 33, 's9': 34, 's10': 35, 's11': 36, 't3': 37, 't4': 38, 't5': 39, 't6': 40, 'ft0': 41, 'ft1': 42, 'ft2': 43, 'ft3': 44,
        'ft4': 45, 'ft5': 46, 'ft6': 47, 'ft7': 48, 'fs0': 49, 'fs1': 50, 'fa0': 51, 'fa1': 52, 'fa2': 53, 'fa3': 54, 'fa4': 55,
        'fa5': 56, 'fa6': 57, 'fa7': 58, 'fs2': 59, 'fs3': 60, 'fs4': 61, 'fs5': 62, 'fs6': 63, 'fs7': 64, 'fs8': 65, 'fs9': 66,
        'fs10': 67, 'fs11': 68, 'ft8': 69, 'ft9': 70, 'ft10': 71, 'ft11': 72, 'x0': 9, 'x1': 10, 'x2': 11, 'x3': 12, 'x4': 13,
        'x5': 14, 'x6': 15, 'x7': 16, 'x8': 17, 'x9': 18, 'x10': 19, 'x11': 20, 'x12': 21, 'x13': 22, 'x14': 23, 'x15': 24, 'x16': 25,
        'x17': 26, 'x18': 27, 'x19': 28, 'x20': 29, 'x21': 30, 'x22': 31, 'x23': 32, 'x24': 33, 'x25': 34, 'x26': 35, 'x27': 36,
        'x28': 37, 'x29': 38, 'x30': 39, 'x31': 40, 'f0': 41, 'f1': 42, 'f2': 43, 'f3': 44, 'f4': 45, 'f5': 46, 'f6': 47, 'f7': 48,
        'f8': 49, 'f9': 50, 'f10': 51, 'f11': 52, 'f12': 53, 'f13': 54, 'f14': 55, 'f15': 56, 'f16': 57, 'f17': 58, 'f18': 59,
        'f19': 60, 'f20': 61, 'f21': 62, 'f22': 63, 'f23': 64, 'f24': 65, 'f25': 66, 'f26': 67, 'f27': 68, 'f28': 69, 'f29': 70,
        'f30': 71, 'f31': 72, 'div': 125, 'divu': 126, 'divuw': 127, 'divw': 128, 'mul': 129, 'mulh': 130, 'mulhsu': 131, 'mulhu': 132,
        'mulw': 133, 'rem': 134, 'remu': 135, 'remuw': 136, 'remw': 137, 'add': 73, 'addi': 74, 'addiw': 75, 'addw': 76, 'and': 77,
        'andi': 78, 'auipc': 79, 'beq': 80, 'bge': 81, 'bgeu': 82, 'blt': 83, 'bltu': 84, 'bne': 85, 'ebreak': 86, 'ecall': 87,
        'fence': 88, 'jal': 89, 'jalr': 90, 'lb': 91, 'lbu': 92, 'ld': 93, 'lh': 94, 'lhu': 95, 'lui': 96, 'lw': 97, 'lwu': 98,
        'or': 99, 'ori': 100, 'sb': 101, 'sd': 102, 'sh': 103, 'sll': 104, 'slli': 105, 'slliw': 106, 'sllw': 107, 'slt': 108,
        'slti': 109, 'sltiu': 110, 'sltu': 111, 'sra': 112, 'srai': 113, 'sraiw': 114, 'sraw': 115, 'srl': 116, 'srli': 117,
        'srliw': 118, 'srlw': 119, 'sub': 120, 'subw': 121, 'sw': 122, 'xor': 123, 'xori': 124}

        self.mnemonic_to_token = {'div': 125, 'divu': 126, 'divuw': 127, 'divw': 128, 'mul': 129, 'mulh': 130, 'mulhsu': 131, 'mulhu': 132,
        'mulw': 133, 'rem': 134, 'remu': 135, 'remuw': 136, 'remw': 137, 'add': 73, 'addi': 74, 'addiw': 75, 'addw': 76, 'and': 77,
        'andi': 78, 'auipc': 79, 'beq': 80, 'bge': 81, 'bgeu': 82, 'blt': 83, 'bltu': 84, 'bne': 85, 'ebreak': 86, 'ecall': 87,
        'fence': 88, 'jal': 89, 'jalr': 90, 'lb': 91, 'lbu': 92, 'ld': 93, 'lh': 94, 'lhu': 95, 'lui': 96, 'lw': 97, 'lwu': 98,
        'or': 99, 'ori': 100, 'sb': 101, 'sd': 102, 'sh': 103, 'sll': 104, 'slli': 105, 'slliw': 106, 'sllw': 107, 'slt': 108,
        'slti': 109, 'sltiu': 110, 'sltu': 111, 'sra': 112, 'srai': 113, 'sraiw': 114, 'sraw': 115, 'srl': 116, 'srli': 117,
        'srliw': 118, 'srlw': 119, 'sub': 120, 'subw': 121, 'sw': 122, 'xor': 123, 'xori': 124}

    def parse_instruction(self, instruction: str) -> Dict:
        """
        将RISC-V指令解析为其组成部分

        Args:
            instruction: RISC-V汇编指令字符串

        Returns:
            包含指令组成部分的字典
        """
        instruction = instruction.strip().lower()

        # 使用正则表达式提取助记符和操作数
        match = re.match(r'([a-z0-9\.]+)\s*(.*)', instruction)
        if not match:
            return {'mnemonic': '<UNK>', 'operands': []}

        mnemonic, operands_str = match.groups()

        # 按逗号分割操作数，处理潜在的空格
        operands = [op.strip() for op in operands_str.split(',')] if operands_str else []

        return {
            'mnemonic': mnemonic,
            'operands': operands
        }

    def get_token_id(self, token: str, token_type: str = None) -> int:
        """
        获取token的ID，确保使用正确的预定义映射

        Args:
            token: 要获取ID的token
            token_type: token的类型，如'mnemonic'或'register'

        Returns:
            token的ID
        """
        token = token.lower()

        # 对于助记符，直接使用助记符到token的映射
        if token_type == 'mnemonic':
            return self.mnemonic_to_token.get(token, 0)

        # 否则使用通用token映射
        return self.token_to_idx.get(token, 0)

    def build_graph(self, basic_block: List[str], encoded_tokens: List[List[int]] = None) -> torch_geometric.data.Data:
        """
        构建RISC-V基本块的图表示

        Args:
            basic_block: RISC-V汇编指令列表
            encoded_tokens: 可选，预编码的token IDs列表，如果提供则优先使用

        Returns:
            表示图的PyTorch Geometric Data对象
        """
        nodes = []  # (type, token)
        edges = []  # (src, dst, type)
        instruction_token_ids = []  # 保存每条指令的token ID

        # 用于跟踪值节点(寄存器、内存等)的映射
        value_nodes = {}  # 将值名称映射到节点索引
        instruction_nodes = []  # 指令节点索引列表

        node_idx = 0

        for instr_idx, instruction in enumerate(basic_block):
            parsed = self.parse_instruction(instruction)
            mnemonic = parsed['mnemonic']
            operands = parsed['operands']

            # 为当前指令获取正确的token ID
            if encoded_tokens is not None and instr_idx < len(encoded_tokens):
                # 使用预编码的token ID（第一个元素是指令token ID）
                token_id = encoded_tokens[instr_idx][0] if encoded_tokens[instr_idx] else 0
            else:
                # 使用预定义映射查找token ID
                token_id = self.get_token_id(mnemonic, 'mnemonic')

            # 保存指令token ID
            instruction_token_ids.append(token_id)

            # 添加指令助记符节点
            mnemonic_node_idx = node_idx
            nodes.append((self.node_types['mnemonic'], token_id))  # 使用正确的token ID
            instruction_nodes.append(mnemonic_node_idx)
            node_idx += 1

            # 添加到前一条指令的结构依赖边
            if instr_idx > 0:
                edges.append((instruction_nodes[instr_idx - 1], mnemonic_node_idx, self.edge_types['structural']))

            # 处理目标操作数(输出)
            if operands:
                dest_operand = operands[0]

                # 检查是否为寄存器
                if dest_operand in self.token_to_idx and (
                        dest_operand.startswith(('x', 'a', 's', 't')) or
                        dest_operand in ('ra', 'sp', 'gp', 'tp', 'fp', 'zero')):
                    # 为目标创建新的寄存器节点
                    dest_node_idx = node_idx
                    nodes.append((self.node_types['register'], self.token_to_idx.get(dest_operand, 0)))
                    node_idx += 1

                    # 添加从指令到寄存器的输出边
                    edges.append((mnemonic_node_idx, dest_node_idx, self.edge_types['output']))

                    # 更新value_nodes以指向这个新节点
                    value_nodes[dest_operand] = dest_node_idx

                # 检查是否为内存存储
                elif '(' in dest_operand and ')' in dest_operand:
                    # 像"sw x1, 8(x2)"这样的内存存储
                    # 提取偏移量和基址寄存器
                    match = re.match(r'(\d+)\(([^\)]+)\)', dest_operand)
                    if match:
                        offset, base_reg = match.groups()

                        # 创建地址节点
                        addr_node_idx = node_idx
                        nodes.append((self.node_types['address'], self.token_to_idx.get('<ADDRESS>', 0)))
                        node_idx += 1

                        # 创建内存节点
                        mem_node_idx = node_idx
                        nodes.append((self.node_types['memory'], self.token_to_idx.get('<CONST>', 0)))
                        node_idx += 1

                        # 添加基址寄存器到地址的边
                        if base_reg in value_nodes:
                            edges.append((value_nodes[base_reg], addr_node_idx, self.edge_types['address_base']))

                        # 添加立即偏移量到地址的边
                        imm_node_idx = node_idx
                        nodes.append((self.node_types['immediate'], self.token_to_idx.get('<CONST>', 0)))
                        node_idx += 1
                        edges.append((imm_node_idx, addr_node_idx, self.edge_types['address_offset']))

                        # 添加从指令到内存的输出边
                        edges.append((mnemonic_node_idx, mem_node_idx, self.edge_types['output']))

            # 处理源操作数(输入)
            for src_idx, src_operand in enumerate(operands[1:], 1):
                # 检查是否为寄存器
                if src_operand in self.token_to_idx and (
                        src_operand.startswith(('x', 'a', 's', 't')) or
                        src_operand in ('ra', 'sp', 'gp', 'tp', 'fp', 'zero')):
                    # 使用现有寄存器节点或创建新节点
                    if src_operand in value_nodes:
                        src_node_idx = value_nodes[src_operand]
                    else:
                        src_node_idx = node_idx
                        nodes.append((self.node_types['register'], self.token_to_idx.get(src_operand, 0)))
                        value_nodes[src_operand] = src_node_idx
                        node_idx += 1

                    # 添加从寄存器到指令的输入边
                    edges.append((src_node_idx, mnemonic_node_idx, self.edge_types['input']))

                # 检查是否为立即数值
                elif src_operand.lstrip('-').isdigit() or (
                        src_operand.startswith('0x') and all(c in '0123456789abcdefABCDEF' for c in src_operand[2:])):
                    # 创建立即数节点
                    imm_node_idx = node_idx
                    nodes.append((self.node_types['immediate'], self.token_to_idx.get('<CONST>', 0)))
                    node_idx += 1

                    # 添加从立即数到指令的输入边
                    edges.append((imm_node_idx, mnemonic_node_idx, self.edge_types['input']))

                # 检查是否为内存加载
                elif '(' in src_operand and ')' in src_operand:
                    # 像"lw x1, 8(x2)"这样的内存加载
                    # 提取偏移量和基址寄存器
                    match = re.match(r'(\d+)\(([^\)]+)\)', src_operand)
                    if match:
                        offset, base_reg = match.groups()

                        # 创建地址节点
                        addr_node_idx = node_idx
                        nodes.append((self.node_types['address'], self.token_to_idx.get('<ADDRESS>', 0)))
                        node_idx += 1

                        # 创建内存节点
                        mem_node_idx = node_idx
                        nodes.append((self.node_types['memory'], self.token_to_idx.get('<CONST>', 0)))
                        node_idx += 1

                        # 添加基址寄存器到地址的边
                        if base_reg in value_nodes:
                            edges.append((value_nodes[base_reg], addr_node_idx, self.edge_types['address_base']))

                        # 添加立即偏移量到地址的边
                        imm_node_idx = node_idx
                        nodes.append((self.node_types['immediate'], self.token_to_idx.get('<CONST>', 0)))
                        node_idx += 1
                        edges.append((imm_node_idx, addr_node_idx, self.edge_types['address_offset']))

                        # 添加从内存到指令的输入边
                        edges.append((mem_node_idx, mnemonic_node_idx, self.edge_types['input']))

        # 创建PyTorch Geometric的张量
        x = torch.zeros((len(nodes), 2), dtype=torch.long)
        edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
        edge_attr = torch.zeros((len(edges), 1), dtype=torch.long)

        # 填充节点特征: 类型和token
        for i, (node_type, token_idx) in enumerate(nodes):
            x[i, 0] = node_type
            x[i, 1] = token_idx

        # 填充边特征
        for i, (src, dst, edge_type) in enumerate(edges):
            edge_index[0, i] = src
            edge_index[1, i] = dst
            edge_attr[i, 0] = edge_type

        # 创建指令节点的掩码
        instruction_mask = torch.zeros(len(nodes), dtype=torch.bool)
        for idx in instruction_nodes:
            instruction_mask[idx] = True

        # 保存指令token IDs
        instruction_token_ids_tensor = torch.tensor(instruction_token_ids, dtype=torch.long)

        # 创建PyTorch Geometric Data对象
        data = torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            instruction_mask=instruction_mask,
            instruction_token_ids=instruction_token_ids_tensor,
            num_nodes=len(nodes)
        )

        return data


"""
DataBatch(x=[78, 2], x_batch=[78], x_ptr=[5], edge_index=[2, 74], edge_attr=[74, 1], instruction_mask=[78], num_nodes=78, batch=[78], ptr=[5])
每个参数的含义

x=[78, 2] 节点特征矩阵，共有78个节点，每个节点有2个特征

第一列表示节点类型（0=指令助记符，1=寄存器，2=立即数，3=内存，4=地址）
第二列表示具体值的编码（例如，指令"add"的token ID）

edge_index=[2, 74] 边连接信息，有74条边  第一行是源节点索引，第二行是目标节点索引

edge_attr=[74, 1] 边属性，形状为[74, 1] 每条边有1个特征，表示边的类型（如输入、输出、结构依赖等）

instruction_mask=[78]  指令节点掩码，长度为78的布尔张量 标识哪些节点是指令助记符节点

batch=[78]和x_batch=[78]  节点批次分配，长度为78 表示每个节点属于哪个样本（图） 例如，batch[i]=2表示第i个节点属于第2个样本

ptr=[5]和x_ptr=[5] 快速索引，长度为样本数+1 可用于确定每个样本的节点范围 例如，第j个样本的节点索引范围是ptr[j]:ptr[j+1]

num_nodes=78 节点总数
"""

class RISCVGraphDataset(Dataset):

    def __init__(self, json_path, cache_dir=None, rebuild_cache=False):

        self.json_path = json_path
        self.cache_dir = cache_dir
        self.graph_encoder = RISCVGraphEncoder()

        cache_file = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            json_filename = os.path.basename(json_path)
            cache_file = os.path.join(cache_dir, f"{os.path.splitext(json_filename)[0]}_graph_cache.pt")

        if cache_file is not None and os.path.exists(cache_file) and not rebuild_cache:
            print(f"从缓存加载图数据: {cache_file}")
            cached_data = torch.load(cache_file)
            self.graphs = cached_data['graphs']
            self.instruction_counts = cached_data['instruction_counts']
            self.throughputs = cached_data['throughputs']
            self.raw_instructions = cached_data.get('raw_instructions', [None] * len(self.graphs))
        else:
            print(f"预处理数据并构建图...")
            # 加载原始数据
            with open(json_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)

            # 预处理所有数据为图结构
            self.graphs = []
            self.instruction_counts = []
            self.throughputs = []
            self.raw_instructions = []

            for i, sample in enumerate(tqdm(self.data, desc="构建图")):
                instructions = sample.get('instructions', [])
                self.raw_instructions.append(instructions)

                if instructions:
                    num_instructions = sample.get('num_instructions', len(instructions))
                    graph = self.graph_encoder.build_graph(instructions)

                    instruction_mask = graph.instruction_mask
                    instruction_nodes = torch.where(instruction_mask)[0]
                    instruction_token_ids = graph.x[instruction_nodes, 1]  # 第二列是token ID
                    graph.instruction_token_ids = instruction_token_ids

                    self.graphs.append(graph)
                    self.instruction_counts.append(num_instructions)
                    self.throughputs.append(sample['throughput'])

            if cache_file is not None:
                print(f"保存图数据到缓存: {cache_file}")
                torch.save({
                    'graphs': self.graphs,
                    'instruction_counts': self.instruction_counts,
                    'throughputs': self.throughputs,
                    'raw_instructions': self.raw_instructions,
                }, cache_file)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):

        return {
            'X': self.graphs[idx],
            'instruction_count': self.instruction_counts[idx],
            'Y': self.throughputs[idx],
        }


if __name__ == '__main__':
    encoder = RISCVGraphEncoder()
    instruction = ['addi	sp,sp,-32','sd	s0,16(sp)','ld	s2,8(a4)','auipc	ra,0x390']
    # for i in instruction:
    #     parsed = encoder.parse_instruction(i)
    #     print(parsed)
    #     token_id = encoder.get_token_id(parsed['mnemonic'], 'mnemonic')
    #     print(token_id)
    graph = encoder.build_graph(instruction) # torch_geometric.data.Data
    print(graph.instruction_token_ids)
