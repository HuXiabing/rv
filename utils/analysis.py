import os
import torch
import random
import numpy as np
from typing import Dict, List, Optional, Any

def analyze_instruction_statistics(instruction_avg_loss: Dict[int, float],
                                   mapping_dict_path: str = "data/vocab.dump",
                                   output_dir: str = "statistics") -> List[float]:
    """
    Analyze instruction type statistics and generate a normalized instruction type vector

    Args:
        instruction_avg_loss: Dictionary of instruction type average losses
        mapping_dict_path: Path to the instruction mapping dictionary
        output_dir: Output directory

    instruction_vec = analyze_instruction_statistics(
            instruction_stats["instruction_avg_loss"],
            mapping_dict_path="data/mapping_dict.dump",
            output_dir=analysis_output_dir
        )

    Returns:
        Normalized instruction type vector

    [shifts_arithmetic_logical_insts,compare_insts,mul_div_insts,load_insts,store_insts]
    """
    os.makedirs(output_dir, exist_ok=True)

    # load mapping dictionary
    mapping_dict = torch.load(mapping_dict_path)
    
    compare_insts = ['slt', 'sltu', 'slti', 'sltiu']
    shifts_arithmetic_logical_insts = ['add', 'addw', 'and', 'sll', 'sllw', 'sra', 'sraw', 'srl', 'srlw', 'sub', 'subw',
                                       'xor','addi', 'addiw', 'andi', 'ori', 'slli', 'slliw', 'srai', 'sraiw', 'srli',
                                       'srliw', 'xori']
    mul_div_insts = ['div', 'divu', 'divuw', 'divw', 'mul', 'mulh', 'mulhsu',
                     'mulhu', 'mulw', 'rem', 'remu', 'remuw', 'remw']
    load_insts = ['lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu']
    store_insts = ['sb', 'sd', 'sh', 'sw']

    new_dict = {key: instruction_avg_loss.get(value, 0.0) for key, value in mapping_dict.items()}
    instruction_loss_path = os.path.join(output_dir, "instruction_loss_details.txt")
    with open(instruction_loss_path, 'w') as file:
        for key, value in new_dict.items():
            file.write(f'{key}: {value}\n')


    shifts_arithmetic_logical_ratio = sum(
        new_dict.get(inst, 0.0) for inst in shifts_arithmetic_logical_insts) + random.uniform(-0.02, 0.02)
    compare_ratio = sum(new_dict.get(inst, 0.0) for inst in compare_insts) + random.uniform(-0.02, 0.02)
    mul_div_ratio = sum(new_dict.get(inst, 0.0) for inst in mul_div_insts) + random.uniform(-0.02, 0.02)
    load_ratio = sum(new_dict.get(inst, 0.0) for inst in load_insts) + random.uniform(-0.02, 0.02)
    store_ratio = sum(new_dict.get(inst, 0.0) for inst in store_insts) + random.uniform(-0.02, 0.02)

    vec= [shifts_arithmetic_logical_ratio, compare_ratio, mul_div_ratio, load_ratio, store_ratio]
    total = sum(vec)
    normalized_vec = [x / total for x in vec]

    # vector_path = os.path.join(output_dir, "instruction_vec.txt")
    # save_vector_to_file(normalized_vec, vector_path)

    return normalized_vec

def analyze_block_length_statistics(block_length_avg_loss: Dict[int, float],
                                    output_dir: str = "statistics") -> Dict[int, float]:
    """
    Analyze basic block length statistics and generate a normalized basic block length dictionary

    Args:
        block_length_avg_loss: Dictionary of basic block length average losses
        output_dir: Output directory

    Returns:
        Normalized basic block length dictionary

    block_dict = analyze_block_length_statistics(
            block_length_stats["block_length_avg_loss"],
            output_dir=analysis_output_dir
        )
    """
    os.makedirs(output_dir, exist_ok=True)

    # find out the maximum basic block length
    max_key = max(block_length_avg_loss.keys())
    ave = 1 / (max(200, max_key) - 2)

    init_dict = {i: ave for i in range(3, max_key + 1)}
    for key, value in init_dict.items():
        if key in block_length_avg_loss:
            block_length_avg_loss[key] += value
        else:
            block_length_avg_loss[key] = value

    total = sum(block_length_avg_loss.values())
    normalized_dict = {key: value / total for key, value in block_length_avg_loss.items()}

    # dict_path = os.path.join(output_dir, "block_dict.txt")
    # save_dict_to_file(normalized_dict, dict_path)

    return normalized_dict

# def process_statistics(instruction_avg_loss_path: str,
#                        block_length_avg_loss_path: str,
#                        mapping_dict_path: str = "data/mapping_dict.dump",
#                        output_dir: str = "./statistics") -> Dict[str, Any]:
#     """
#     处理指令类型和基本块长度的统计数据
#
#     Args:
#         instruction_avg_loss_path: 指令类型平均损失文件路径
#         block_length_avg_loss_path: 基本块长度平均损失文件路径
#         mapping_dict_path: 指令映射字典路径
#         output_dir: 输出目录
#
#     Returns:
#         包含分析结果的字典
#     """
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 加载统计数据
#     instruction_avg_loss = load_dict_from_file(instruction_avg_loss_path)
#     block_length_avg_loss = load_dict_from_file(block_length_avg_loss_path)
#
#     # 分析指令类型统计
#     instruction_vec = analyze_instruction_statistics(
#         instruction_avg_loss,
#         mapping_dict_path,
#         output_dir
#     )
#
#     # 分析基本块长度统计
#     block_dict = analyze_block_length_statistics(
#         block_length_avg_loss,
#         output_dir
#     )
#
#     return {
#         "instruction_vec": instruction_vec,
#         "block_dict": block_dict
#     }

def load_dict_from_file(file_path: str) -> Dict[int, float]:
    """
    从文本文件加载字典数据

    Args:
        file_path: 字典文件路径，每行格式为 "key: value"

    Returns:
        加载的字典，键为整数，值为浮点数
    """
    loss = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            loss[int(key)] = float(value)
    return loss


def save_dict_to_file(data: Dict[int, float], file_path: str) -> None:
    """
    将字典保存到文本文件

    Args:
        data: 要保存的字典，键为整数，值为浮点数
        file_path: 保存的文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        for key, value in data.items():
            file.write(f'{key}: {value}\n')
