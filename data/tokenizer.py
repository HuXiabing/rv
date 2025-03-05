import re
import torch
import os
from typing import List, Dict, Any, Optional
from collections import Counter

class RISCVTokenizer:
    """RISC-V指令的分词器，使用自定义分词和词汇表"""

    def __init__(self,
                 vocab_size: int = 300,
                 max_instr_length: int = 8):
        """
        初始化RISC-V分词器

        Args:
            vocab_size: 词汇表大小的上限（不实际使用，为了兼容性保留）
            max_instr_length: 每条指令的最大标记数，固定为8
        """
        self.vocab_size = vocab_size
        self.max_instr_length = max_instr_length  # 固定为8

        # 加载预定义的词汇表
        self._load_predefined_dict()

        # 定义隐式寻址指令列表
        self.implicit_list = ['fld', 'flw', 'fsd', 'fsw', 'lb', 'lbu', 'ld', 'lh', 'lhu', 'lw',
                              'lwu', 'sb', 'sd', 'sh', 'sw']

    def _load_predefined_dict(self):
        """加载预定义的词汇表"""
        # 使用字典定义或加载预保存的字典文件
        # 尝试直接从代码文件所在目录加载
        dict_path = os.path.join(os.path.dirname(__file__), 'mapping_dict.dump')
        if os.path.exists(dict_path):
            self.vocab = torch.load(dict_path)
        else:
            # 如果找不到文件，手动构建词汇表
            self._build_vocab_manually()

        # 反向映射，用于解码
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.current_vocab_size = len(self.vocab)

    def _build_vocab_manually(self):
        """手动构建词汇表（如果无法加载预定义文件）"""
        # 特殊标记
        special_token = {'<PAD>': 0, '<BLOCK_START>': 1, '<BLOCK_END>': 2, '<ADDRESS>': 3,
                         '<E>': 4, '<D>': 5, '<S>': 6, '<CONST>': 7, '<CSR>': 8}

        _xregs_abi = {'zero': 9, 'ra': 10, 'sp': 11, 'gp': 12, 'tp': 13, 't0': 14, 't1': 15, 't2': 16,
                      's0': 17, 's1': 18, 'a0': 19, 'a1': 20, 'a2': 21, 'a3': 22, 'a4': 23, 'a5': 24,
                      'a6': 25, 'a7': 26, 's2': 27, 's3': 28, 's4': 29, 's5': 30, 's6': 31, 's7': 32,
                      's8': 33, 's9': 34, 's10': 35, 's11': 36, 't3': 37, 't4': 38, 't5': 39, 't6': 40}

        _xregs_numeric_list = [f'x{i}' for i in range(32)]
        _fregs_numeric_list = [f'f{i}' for i in range(32)]

        shift = 9
        _xregs_numeric = {_xregs_numeric_list[i]: i + shift for i in range(len(_xregs_numeric_list))}
        shift = 41
        _fregs_numeric = {_fregs_numeric_list[i]: i + shift for i in range(len(_fregs_numeric_list))}

        # _xregs_abi_list = [
        #     "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
        #     "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
        #     "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
        #     "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"
        # ]
        # _xregs_abi = { _xregs_abi_list[i] : i + shift for i in range(len(_xregs_abi_list))}

        _fregs_abi = {'ft0': 41, 'ft1': 42, 'ft2': 43, 'ft3': 44, 'ft4': 45, 'ft5': 46, 'ft6': 47, 'fs0': 48, 'fs1': 49,
                      'fa0': 50, 'fa1': 51, \
                      'fa2': 52, 'fa3': 53, 'fa4': 54, 'fa5': 55, 'fa6': 56, 'fa7': 57, 'fs2': 58, 'fs3': 59, 'fs4': 60,
                      'fs5': 61, 'fs6': 62, \
                      'fs7': 63, 'fs8': 64, 'fs9': 65, 'fs10': 66, 'fs11': 67, 'ft8': 68, 'ft9': 69, 'ft10': 70,
                      'ft11': 71}

        # shift = 41
        # _fregs_abi_list = [
        #               'ft0', 'ft1', 'ft2', 'ft3', 'ft4', 'ft5', 'ft6', 'ft7',
        #               'fs0', 'fs1', 'fa0', 'fa1', 'fa2', 'fa3', 'fa4', 'fa5',
        #               'fa6', 'fa7', 'fs2', 'fs3', 'fs4', 'fs5', 'fs6', 'fs7',
        #               'fs8', 'fs9', 'fs10', 'fs11', 'ft8', 'ft9', 'ft10', 'ft11']
        # _fregs_abi = { _fregs_abi_list[i] : i + shift for i in range(len(_fregs_abi_list))}

        _rv32I = {'add': 72, 'addi': 73, 'sub': 223, 'lui': 188, 'auipc': 96, \
                  'sll': 207, 'slli': 208, 'srl': 219, 'srli': 220, 'sra': 215, 'srai': 216, \
                  'slt': 211, 'slti': 212, 'sltiu': 213, 'sltu': 214, \
                  'and': 94, 'andi': 95, 'or': 196, 'ori': 197, 'xor': 226, 'xori': 227, \
                  'beq': 97, 'bge': 98, 'bgeu': 99, 'blt': 100, 'bltu': 101, 'bne': 102, \
                  'lb': 181, 'lbu': 182, 'lh': 184, 'lhu': 185, 'lw': 189, \
                  'sb': 202, 'sh': 206, 'sw': 225, \
                  'jal': 179, 'jalr': 180, 'ebreak': 113, 'ecall': 114
                  }

        _rv32A = {'amoadd.w': 77, 'amoand.w': 79, 'amomax.w': 81, 'amomaxu.w': 83, 'amomin.w': 85, 'amominu.w': 87, \
                  'amoor.w': 89, 'amoswap.w': 91, 'amoxor.w': 93, 'lr.w': 187, 'sc.w': 204}

        _rv32M = {'mul': 191, 'mulh': 192, 'mulhsu': 193, 'mulhu': 194, 'div': 109, 'divu': 110, 'rem': 198,
                  'remu': 199}

        _rv32FD = {'fadd.d': 115, 'fadd.s': 116, 'fclass.d': 117, 'fclass.s': 118, 'fcvt.d.s': 121, 'fcvt.d.w': 122, \
                   'fcvt.d.wu': 123, 'fcvt.s.d': 128, 'fcvt.s.w': 131, 'fcvt.s.wu': 132, 'fcvt.w.d': 133,
                   'fcvt.w.s': 134, 'fcvt.wu.d': 135, 'fcvt.wu.s': 136, \
                   'fdiv.d': 137, 'fdiv.s': 138, 'fence': 139, 'fence.i': 140, 'feq.d': 141, 'feq.s': 142, 'fld': 143,
                   'fle.d': 144, 'fle.s': 145, \
                   'flt.d': 146, 'flt.s': 147, 'fsw': 178, 'flw': 148, 'fmadd.d': 149, 'fmadd.s': 150, 'fmax.d': 151,
                   'fmax.s': 152, 'fmin.d': 153, \
                   'fmin.s': 154, 'fmsub.d': 155, 'fmsub.s': 156, 'fmul.d': 157, 'fmul.s': 158, 'fmv.w.x': 160, \
                   'fmv.x.w': 162, 'fnmadd.d': 163, 'fnmadd.s': 164, 'fnmsub.d': 165, 'fnmsub.s': 166, 'fsd': 167,
                   'fsgnj.d': 168, 'fsgnj.s': 169, \
                   'fsgnjn.d': 170, 'fsgnjn.s': 171, 'fsgnjx.d': 172, 'fsgnjx.s': 173, 'fsqrt.d': 174, 'fsqrt.s': 175,
                   'fsub.d': 176, 'fsub.s': 177}

        _rv64I = {'addiw': 74, 'addw': 75, 'subw': 224, \
                  'srliw': 221, 'srlw': 222, 'slliw': 209, 'sllw': 210, 'sraiw': 217, 'sraw': 218, \
                  'lwu': 190, 'ld': 183, 'sd': 205
                  }

        _rv64A = {'amoadd.d': 76, 'amoand.d': 78, 'amomax.d': 80, 'amomaxu.d': 82, 'amomin.d': 84, 'amominu.d': 86, \
                  'amoor.d': 88, 'amoswap.d': 90, 'amoxor.d': 92, 'lr.d': 186, 'sc.d': 203}

        _rv64M = {'mulw': 195, 'divw': 112, 'divuw': 111, 'remw': 201, 'remuw': 200}

        _rv64FD = {'fmv.d.x': 159, 'fmv.x.d': 161, 'fcvt.s.l': 129, 'fcvt.s.lu': 130, 'fcvt.lu.d': 126,
                   'fcvt.lu.s': 127, \
                   'fcvt.l.d': 124, 'fcvt.l.s': 125, 'fcvt.d.l': 119, 'fcvt.d.lu': 120}

        _CSR = {'csrrc': 103, 'csrrci': 104, 'csrrs': 105, 'csrrsi': 106, 'csrrw': 107, 'csrrwi': 108}

        _rv32 = {**_rv32A, **_rv32M, **_rv32I, **_rv32FD, **_CSR}

        _rv64 = {**_rv32, **_rv64A, **_rv64M, **_rv64I, **_rv64FD}

        self.vocab ={**special_token, **_xregs_abi, **_fregs_abi, **_rv64, **_xregs_numeric, **_fregs_numeric}

    def separate_disp_and_reg(self, s):
        """分离内存地址中的偏移量和寄存器"""
        match = re.match(r'-?\d+\((\w+)\)', s)
        if match:
            full_match = match.group(0)
            reg = match.group(1)
            disp_str = full_match[:-(len(reg) + 2)]
            disp = int(disp_str)
            return disp, reg
        else:
            raise ValueError("The string does not match the expected format.")

    def is_numeric(self, s):
        """检查字符串是否为数字"""
        pattern = re.compile(r'^-?\d+(\.\d+)?$')
        return bool(pattern.match(s))

    def is_address_string(self, s):
        """检查字符串是否为地址格式"""
        pattern = re.compile(r'^0x[0-9a-fA-F]+$')
        return bool(pattern.match(s))

    def zero_reg(self, instr):
        """处理无操作数指令"""
        return [instr[0], '<E>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']

    def one_reg_num(self, instr):
        """处理单个寄存器或数字的指令"""
        pattern = r'^-?\d+$'
        if re.match(pattern, instr[1]):  # "j, -1666"
            return [instr[0], '<CONST>', '<E>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
        else:
            return [instr[0], instr[1], '<E>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']

    def two_reg(self, instr):
        """处理两个操作数的指令"""
        if instr[0] in self.implicit_list:
            _, rs = self.separate_disp_and_reg(instr[-1])
            return [instr[0], '<D>', instr[1], '<S>', rs, '<CONST>', '<E>', '<PAD>']
        elif self.is_numeric(instr[-1]):
            return [instr[0], '<D>', instr[1], '<S>', '<CONST>', '<E>', '<PAD>', '<PAD>']
        elif self.is_address_string(instr[-1]):
            return [instr[0], '<D>', instr[1], '<S>', '<ADDRESS>', '<E>', '<PAD>', '<PAD>']
        else:
            return [instr[0], '<D>', instr[1], '<S>', instr[2], '<E>', '<PAD>', '<PAD>']

    def three_reg(self, instr):
        """处理三个操作数的指令"""
        if instr[1] == 'csr':
            if self.is_numeric(instr[-1]):
                return [instr[0], '<D>', instr[1], '<S>', '<CSR>', '<CONST>', '<E>', '<PAD>']
            elif self.is_address_string(instr[-1]):
                return [instr[0], '<D>', instr[1], '<S>', '<CSR>', '<ADDRESS>', '<E>', '<PAD>']
            else:
                return [instr[0], '<D>', instr[1], '<S>', '<CSR>', instr[-1], '<E>', '<PAD>']
        elif self.is_numeric(instr[-1]):
            return [instr[0], '<D>', instr[1], '<S>', instr[2], '<CONST>', '<E>', '<PAD>']
        elif self.is_address_string(instr[-1]):
            return [instr[0], '<D>', instr[1], '<S>', instr[2], '<ADDRESS>', '<E>', '<PAD>']
        else:
            return [instr[0], '<D>', instr[1], '<S>', instr[2], instr[3], '<E>', '<PAD>']

    def four_reg(self, instr):
        """处理四个操作数的指令"""
        return [instr[0], '<D>', instr[1], '<S>', instr[2], instr[3], instr[4], '<E>']

    def tokenize_instruction(self, instruction: str) -> List[str]:
        """
        将RISC-V指令拆分为令牌列表

        Args:
            instruction: RISC-V汇编指令字符串

        Returns:
            分词后的标记列表
        """
        pattern = r'[^ ,\t]+'  # 匹配非空格、非逗号和非制表符的内容
        instr = re.findall(pattern, instruction)

        instr_len = len(instr)
        if instr_len == 1:
            tokenized = self.zero_reg(instr)
        elif instr_len == 2:
            tokenized = self.one_reg_num(instr)
        elif instr_len == 3:
            tokenized = self.two_reg(instr)
        elif instr_len == 4:
            tokenized = self.three_reg(instr)
        elif instr_len == 5:
            tokenized = self.four_reg(instr)
        else:
            raise ValueError(f"Invalid instruction format: {instruction}")

        return tokenized

    def encode_instruction(self, instruction: str) -> List[int]:
        """
        将单条指令编码为token ID序列

        Args:
            instruction: RISC-V指令字符串

        Returns:
            编码后的token ID列表
        """
        tokens = self.tokenize_instruction(instruction)
        encoded = [self.vocab.get(token, self.vocab.get('<PAD>', 0)) for token in tokens]

        # 不需要截断或填充，因为我们固定使用8个令牌
        return encoded

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        将token ID序列解码为令牌列表

        Args:
            token_ids: token ID列表

        Returns:
            解码后的令牌列表
        """
        return [self.inverse_vocab.get(tid, '<UNK>') for tid in token_ids]

    def tokenized_bb(self, basic_block: str) -> List[List[str]]:
        """
        将基本块分词为令牌列表的列表

        Args:
            basic_block: 基本块字符串，指令以换行符分隔

        Returns:
            分词后的标记列表的列表
        """
        bb_tokens = []
        pattern = r'[^ ,\t]+'
        bb = [re.findall(pattern, s) for s in basic_block.strip().split("\\n")]

        for instruction in bb:
            instr_len = len(instruction)
            if instr_len == 1:
                tokenized = self.zero_reg(instruction)
            elif instr_len == 2:
                tokenized = self.one_reg_num(instruction)
            elif instr_len == 3:
                tokenized = self.two_reg(instruction)
            elif instr_len == 4:
                tokenized = self.three_reg(instruction)
            elif instr_len == 5:
                tokenized = self.four_reg(instruction)
            else:
                raise ValueError(f"Invalid instruction format: {instruction}")

            bb_tokens.append(tokenized)

        return bb_tokens