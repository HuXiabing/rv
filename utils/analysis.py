import os
import torch
import random
import numpy as np
from typing import Dict, List, Optional, Any

def analyze_instruction_statistics(instruction_avg_loss: Dict[int, float],
                                   mapping_dict_path: str = "data/mapping_dict.dump",
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
    if os.path.exists(mapping_dict_path):
        mapping_dict = torch.load(mapping_dict_path)
    else:
        mapping_dict = {'<PAD>': 0, '<BLOCK_START>': 1, '<BLOCK_END>': 2, '<ADDRESS>': 3, '<E>': 4, '<D>': 5, '<S>': 6, '<CONST>': 7, '<CSR>': 8,
                        'zero': 9, 'ra': 10, 'sp': 11, 'gp': 12, 'tp': 13, 't0': 14, 't1': 15, 't2': 16, 's0': 17, 's1': 18, 'a0': 19, 'a1': 20, 'a2': 21, 'a3': 22, 'a4': 23, 'a5': 24, 'a6': 25, 'a7': 26, 's2': 27, 's3': 28, 's4': 29, 's5': 30, 's6': 31, 's7': 32, 's8': 33, 's9': 34, 's10': 35, 's11': 36, 't3': 37, 't4': 38, 't5': 39, 't6': 40, 'ft0': 41, 'ft1': 42, 'ft2': 43, 'ft3': 44, 'ft4': 45, 'ft5': 46, 'ft6': 47, 'fs0': 48, 'fs1': 49, 'fa0': 50, 'fa1': 51, 'fa2': 52, 'fa3': 53, 'fa4': 54, 'fa5': 55, 'fa6': 56, 'fa7': 57,'fs2': 58, 'fs2': 59, 'fs3': 60, 'fs4': 61, 'fs5': 62, 'fs6': 63, 'fs7': 64, 'fs8': 65, 'fs9': 66, 'fs10': 67, 'fs11': 68, 'ft8': 69, 'ft9': 70, 'ft10': 71, 'ft11': 72,
                        'amoadd.w': 77, 'amoand.w': 79, 'amomax.w': 81, 'amomaxu.w': 83, 'amomin.w': 85, 'amominu.w': 87, 'amoor.w': 89, 'amoswap.w': 91, 'amoxor.w': 93, 'lr.w': 187, 'sc.w': 204, 'mul': 191, 'mulh': 192, 'mulhsu': 193, 'mulhu': 194, 'div': 109, 'divu': 110, 'rem': 198, 'remu': 199, 'add': 72, 'addi': 73, 'sub': 223, 'lui': 188, 'auipc': 96, 'sll': 207, 'slli': 208, 'srl': 219, 'srli': 220, 'sra': 215, 'srai': 216, 'slt': 211, 'slti': 212, 'sltiu': 213, 'sltu': 214, 'and': 94, 'andi': 95, 'or': 196, 'ori': 197, 'xor': 226, 'xori': 227, 'beq': 97, 'bge': 98, 'bgeu': 99, 'blt': 100, 'bltu': 101, 'bne': 102, 'lb': 181, 'lbu': 182, 'lh': 184, 'lhu': 185, 'lw': 189, 'sb': 202, 'sh': 206, 'sw': 225, 'jal': 179, 'jalr': 180, 'ebreak': 113, 'ecall': 114, 'fadd.d': 115, 'fadd.s': 116, 'fclass.d': 117, 'fclass.s': 118, 'fcvt.d.s': 121, 'fcvt.d.w': 122, 'fcvt.d.wu': 123, 'fcvt.s.d': 128, 'fcvt.s.w': 131, 'fcvt.s.wu': 132, 'fcvt.w.d': 133, 'fcvt.w.s': 134, 'fcvt.wu.d': 135, 'fcvt.wu.s': 136, 'fdiv.d': 137, 'fdiv.s': 138, 'fence': 139, 'fence.i': 140, 'feq.d': 141, 'feq.s': 142, 'fld': 143, 'fle.d': 144, 'fle.s': 145, 'flt.d': 146, 'flt.s': 147, 'fsw': 178, 'flw': 148, 'fmadd.d': 149, 'fmadd.s': 150, 'fmax.d': 151, 'fmax.s': 152, 'fmin.d': 153, 'fmin.s': 154, 'fmsub.d': 155, 'fmsub.s': 156, 'fmul.d': 157, 'fmul.s': 158, 'fmv.w.x': 160, 'fmv.x.w': 162, 'fnmadd.d': 163, 'fnmadd.s': 164, 'fnmsub.d': 165, 'fnmsub.s': 166, 'fsd': 167, 'fsgnj.d': 168, 'fsgnj.s': 169, 'fsgnjn.d': 170, 'fsgnjn.s': 171, 'fsgnjx.d': 172, 'fsgnjx.s': 173, 'fsqrt.d': 174, 'fsqrt.s': 175, 'fsub.d': 176, 'fsub.s': 177, 'csrrc': 103, 'csrrci': 104, 'csrrs': 105, 'csrrsi': 106, 'csrrw': 107, 'csrrwi': 108, 'amoadd.d': 76, 'amoand.d': 78, 'amomax.d': 80, 'amomaxu.d': 82, 'amomin.d': 84, 'amominu.d': 86, 'amoor.d': 88, 'amoswap.d': 90, 'amoxor.d': 92, 'lr.d': 186, 'sc.d': 203, 'mulw': 195, 'divw': 112, 'divuw': 111, 'remw': 201, 'remuw': 200, 'addiw': 74, 'addw': 75, 'subw': 224, 'srliw': 221, 'srlw': 222, 'slliw': 209, 'sllw': 210, 'sraiw': 217, 'sraw': 218, 'lwu': 190, 'ld': 183, 'sd': 205, 'fmv.d.x': 159, 'fmv.x.d': 161, 'fcvt.s.l': 129, 'fcvt.s.lu': 130, 'fcvt.lu.d': 126, 'fcvt.lu.s': 127, 'fcvt.l.d': 124, 'fcvt.l.s': 125, 'fcvt.d.l': 119, 'fcvt.d.lu': 120,
                        'x0': 9, 'x1': 10, 'x2': 11, 'x3': 12, 'x4': 13, 'x5': 14, 'x6': 15, 'x7': 16, 'x8': 17, 'x9': 18, 'x10': 19, 'x11': 20, 'x12': 21, 'x13': 22, 'x14': 23, 'x15': 24, 'x16': 25, 'x17': 26, 'x18': 27, 'x19': 28, 'x20': 29, 'x21': 30, 'x22': 31, 'x23': 32, 'x24': 33, 'x25': 34, 'x26': 35, 'x27': 36, 'x28': 37, 'x29': 38, 'x30': 39, 'x31': 40, 'f0': 41, 'f1': 42, 'f2': 43, 'f3': 44, 'f4': 45, 'f5': 46, 'f6': 47, 'f7': 48, 'f8': 49, 'f9': 50, 'f10': 51, 'f11': 52, 'f12': 53, 'f13': 54, 'f14': 55, 'f15': 56, 'f16': 57, 'f17': 58, 'f18': 59, 'f19': 60, 'f20': 61, 'f21': 62, 'f22': 63, 'f23': 64, 'f24': 65, 'f25': 66, 'f26': 67, 'f27': 68, 'f28': 69, 'f29': 70, 'f30': 71, 'f31': 72}

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
    ave = 1 / (max_key - 2)

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


def save_vector_to_file(vector: List[float], file_path: str) -> None:
    """
    将向量保存到文本文件

    Args:
        vector: 要保存的向量
        file_path: 保存的文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(f'{vector}\n')
