import re
import torch
import os
from typing import List, Dict, Any, Optional
from collections import Counter

class RISCVTokenizer:
    """RISC-V tokenizer with custom tokenization and vocabulary"""

    def __init__(self,
                 vocab_size: int = 300,
                 max_instr_length: int = 8):
        """
        Args:
            vocab_size: the upper limit of vocabulary size (not actually used, reserved for compatibility)
            max_instr_length: maximum number of tokens for each instruction, fixed to 8
        """
        self.vocab_size = vocab_size
        self.max_instr_length = max_instr_length

        self._load_predefined_dict()
        self.implicit_list = ['fld', 'flw', 'fsd', 'fsw', 'lb', 'lbu', 'ld', 'lh', 'lhu', 'lw',
                              'lwu', 'sb', 'sd', 'sh', 'sw']

    def _load_predefined_dict(self):
        """ load predefined vocabulary """
        dict_path = os.path.join(os.path.dirname(__file__), 'mapping_dict.dump')
        if os.path.exists(dict_path):
            self.vocab = torch.load(dict_path)
        else:
            # if the predefined dict does not exist, build it manually
            self._build_vocab_manually()

        # inverse the vocabulary to decode token IDs
        # self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.current_vocab_size = len(self.vocab)

    def _build_vocab_manually(self):

        special_token = {'<PAD>': 0, '<BLOCK_START>': 1, '<BLOCK_END>': 2, '<ADDRESS>': 3,
                         '<E>': 4, '<D>': 5, '<S>': 6, '<CONST>': 7, '<CSR>': 8}

        _xregs_abi_list = ['zero', 'ra', 'sp', 'gp', 'tp', 't0', 't1', 't2',
                      's0', 's1', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5',
                      'a6', 'a7', 's2', 's3', 's4', 's5', 's6', 's7',
                      's8', 's9', 's10', 's11', 't3', 't4', 't5', 't6']

        _fregs_abi_list = ['ft0', 'ft1', 'ft2', 'ft3', 'ft4', 'ft5', 'ft6', 'ft7',
                      'fs0', 'fs1', 'fa0', 'fa1', 'fa2', 'fa3', 'fa4', 'fa5',
                      'fa6', 'fa7', 'fs2', 'fs3', 'fs4', 'fs5', 'fs6', 'fs7',
                      'fs8', 'fs9', 'fs10', 'fs11', 'ft8', 'ft9', 'ft10', 'ft11']

        _xregs_numeric_list = [f'x{i}' for i in range(32)]
        _fregs_numeric_list = [f'f{i}' for i in range(32)]

        _xregs_numeric = {_xregs_numeric_list[i]: i + 9 for i in range(len(_xregs_numeric_list))}
        _xregs_abi = { _xregs_abi_list[i] : i + 9 for i in range(len(_xregs_abi_list))}
        _fregs_numeric = {_fregs_numeric_list[i]: i + 41 for i in range(len(_fregs_numeric_list))}
        _fregs_abi = { _fregs_abi_list[i] : i + 41 for i in range(len(_fregs_abi_list))}

        _rv64I_list = ["add", "addi", "addiw", "addw", "and", "andi", "auipc", "beq", "bge",
                    "bgeu", "blt", "bltu", "bne", "ebreak", "ecall", "fence", "jal", "jalr",
                    "lb", "lbu", "ld", "lh", "lhu", "lui", "lw", "lwu", "or", "ori", "sb",
                    "sd", "sh", "sll", "slli", "slliw", "sllw", "slt", "slti", "sltiu",
                    "sltu", "sra", "srai", "sraiw", "sraw", "srl", "srli", "srliw", "srlw",
                    "sub", "subw", "sw", "xor", "xori"] #52
        _rv64I = {_rv64I_list[i]: i + 73 for i in range(len(_rv64I_list))}

        _rv64A_list =["amoadd.d", "amoadd.w", "amoand.d", "amoand.w", "amomax.d", "amomax.w",
                    "amomaxu.d", "amomaxu.w", "amomin.d", "amomin.w", "amominu.d", "amominu.w",
                    "amoor.d", "amoor.w", "amoswap.d", "amoswap.w", "amoxor.d", "amoxor.w",
                    "lr.d", "lr.w", "sc.d", "sc.w"] #22
        _rv64A = {_rv64A_list[i]: i + 125 for i in range(len(_rv64A_list))}

        _rv64M_list = ["div", "divu", "divuw", "divw", "mul", "mulh", "mulhsu", "mulhu", "mulw",
                    "rem", "remu", "remuw", "remw"] #13
        _rv64M = {_rv64M_list[i]: i + 147 for i in range(len(_rv64M_list))}

        _rv64FD_list = ["fadd.d", "fadd.s", "fclass.d", "fclass.s", "fcvt.d.l", "fcvt.d.lu", "fcvt.d.s",
                    "fcvt.d.w", "fcvt.d.wu", "fcvt.l.d", "fcvt.l.s", "fcvt.lu.d", "fcvt.lu.s",
                    "fcvt.s.d", "fcvt.s.l", "fcvt.s.lu", "fcvt.s.w", "fcvt.s.wu", "fcvt.w.d",
                    "fcvt.w.s", "fcvt.wu.d", "fcvt.wu.s", "fdiv.d", "fdiv.s", "feq.d", "feq.s",
                    "fld", "fle.d", "fle.s", "flt.d", "flt.s", "flw", "fmadd.d", "fmadd.s",
                    "fmax.d", "fmax.s", "fmin.d", "fmin.s", "fmsub.d", "fmsub.s", "fmul.d",
                    "fmul.s", "fmv.d.x", "fmv.w.x", "fmv.x.d", "fmv.x.w", "fnmadd.d", "fnmadd.s",
                    "fnmsub.d", "fnmsub.s", "fsd", "fsgnj.d", "fsgnj.s", "fsgnjn.d", "fsgnjn.s",
                    "fsgnjx.d", "fsgnjx.s", "fsqrt.d", "fsqrt.s", "fsub.d", "fsub.s", "fsw"] #62
        _rv64FD = {_rv64FD_list[i]: i + 160 for i in range(len(_rv64FD_list))}

        _CSR_list = ["csrrc", "csrrci", "csrrs", "csrrsi", "csrrw", "csrrwi", "fence.i"] #7
        _CSR = {_CSR_list[i]: i + 222 for i in range(len(_CSR_list))}

        _rv64 = {**_rv64A, **_rv64M, **_rv64I, **_rv64FD, **_CSR}

        self.vocab ={**special_token, **_xregs_abi, **_fregs_abi, **_xregs_numeric, **_fregs_numeric, **_rv64}
        torch.save(self.vocab, "data/vocab.dump")
        """
        {'<PAD>': 0, '<BLOCK_START>': 1, '<BLOCK_END>': 2, '<ADDRESS>': 3, '<E>': 4, '<D>': 5, '<S>': 6, '<CONST>': 7, '<CSR>': 8, 'zero': 9, 'ra': 10, 'sp': 11, 'gp': 12, 'tp': 13, 't0': 14, 't1': 15, 't2': 16
, 's0': 17, 's1': 18, 'a0': 19, 'a1': 20, 'a2': 21, 'a3': 22, 'a4': 23, 'a5': 24, 'a6': 25, 'a7': 26, 's2': 27, 's3': 28, 's4': 29, 's5': 30, 's6': 31, 's7': 32, 's8': 33, 's9': 34, 's10': 35, 's11': 36
, 't3': 37, 't4': 38, 't5': 39, 't6': 40, 'ft0': 41, 'ft1': 42, 'ft2': 43, 'ft3': 44, 'ft4': 45, 'ft5': 46, 'ft6': 47, 'ft7': 48, 'fs0': 49, 'fs1': 50, 'fa0': 51, 'fa1': 52, 'fa2': 53, 'fa3': 54, 'fa4':
 55, 'fa5': 56, 'fa6': 57, 'fa7': 58, 'fs2': 59, 'fs3': 60, 'fs4': 61, 'fs5': 62, 'fs6': 63, 'fs7': 64, 'fs8': 65, 'fs9': 66, 'fs10': 67, 'fs11': 68, 'ft8': 69, 'ft9': 70, 'ft10': 71, 'ft11': 72, 'x0': 
9, 'x1': 10, 'x2': 11, 'x3': 12, 'x4': 13, 'x5': 14, 'x6': 15, 'x7': 16, 'x8': 17, 'x9': 18, 'x10': 19, 'x11': 20, 'x12': 21, 'x13': 22, 'x14': 23, 'x15': 24, 'x16': 25, 'x17': 26, 'x18': 27, 'x19': 28,
 'x20': 29, 'x21': 30, 'x22': 31, 'x23': 32, 'x24': 33, 'x25': 34, 'x26': 35, 'x27': 36, 'x28': 37, 'x29': 38, 'x30': 39, 'x31': 40, 'f0': 41, 'f1': 42, 'f2': 43, 'f3': 44, 'f4': 45, 'f5': 46, 'f6': 47,
 'f7': 48, 'f8': 49, 'f9': 50, 'f10': 51, 'f11': 52, 'f12': 53, 'f13': 54, 'f14': 55, 'f15': 56, 'f16': 57, 'f17': 58, 'f18': 59, 'f19': 60, 'f20': 61, 'f21': 62, 'f22': 63, 'f23': 64, 'f24': 65, 'f25':
 66, 'f26': 67, 'f27': 68, 'f28': 69, 'f29': 70, 'f30': 71, 'f31': 72, 'amoadd.d': 125, 'amoadd.w': 126, 'amoand.d': 127, 'amoand.w': 128, 'amomax.d': 129, 'amomax.w': 130, 'amomaxu.d': 131, 'amomaxu.w'
: 132, 'amomin.d': 133, 'amomin.w': 134, 'amominu.d': 135, 'amominu.w': 136, 'amoor.d': 137, 'amoor.w': 138, 'amoswap.d': 139, 'amoswap.w': 140, 'amoxor.d': 141, 'amoxor.w': 142, 'lr.d': 143, 'lr.w': 14
4, 'sc.d': 145, 'sc.w': 146, 'div': 147, 'divu': 148, 'divuw': 149, 'divw': 150, 'mul': 151, 'mulh': 152, 'mulhsu': 153, 'mulhu': 154, 'mulw': 155, 'rem': 156, 'remu': 157, 'remuw': 158, 'remw': 159, 'a
dd': 73, 'addi': 74, 'addiw': 75, 'addw': 76, 'and': 77, 'andi': 78, 'auipc': 79, 'beq': 80, 'bge': 81, 'bgeu': 82, 'blt': 83, 'bltu': 84, 'bne': 85, 'ebreak': 86, 'ecall': 87, 'fence': 88, 'jal': 89, '
jalr': 90, 'lb': 91, 'lbu': 92, 'ld': 93, 'lh': 94, 'lhu': 95, 'lui': 96, 'lw': 97, 'lwu': 98, 'or': 99, 'ori': 100, 'sb': 101, 'sd': 102, 'sh': 103, 'sll': 104, 'slli': 105, 'slliw': 106, 'sllw': 107, 
'slt': 108, 'slti': 109, 'sltiu': 110, 'sltu': 111, 'sra': 112, 'srai': 113, 'sraiw': 114, 'sraw': 115, 'srl': 116, 'srli': 117, 'srliw': 118, 'srlw': 119, 'sub': 120, 'subw': 121, 'sw': 122, 'xor': 123
, 'xori': 124, 'fadd.d': 160, 'fadd.s': 161, 'fclass.d': 162, 'fclass.s': 163, 'fcvt.d.l': 164, 'fcvt.d.lu': 165, 'fcvt.d.s': 166, 'fcvt.d.w': 167, 'fcvt.d.wu': 168, 'fcvt.l.d': 169, 'fcvt.l.s': 170, 'f
cvt.lu.d': 171, 'fcvt.lu.s': 172, 'fcvt.s.d': 173, 'fcvt.s.l': 174, 'fcvt.s.lu': 175, 'fcvt.s.w': 176, 'fcvt.s.wu': 177, 'fcvt.w.d': 178, 'fcvt.w.s': 179, 'fcvt.wu.d': 180, 'fcvt.wu.s': 181, 'fdiv.d': 1
82, 'fdiv.s': 183, 'feq.d': 184, 'feq.s': 185, 'fld': 186, 'fle.d': 187, 'fle.s': 188, 'flt.d': 189, 'flt.s': 190, 'flw': 191, 'fmadd.d': 192, 'fmadd.s': 193, 'fmax.d': 194, 'fmax.s': 195, 'fmin.d': 196
, 'fmin.s': 197, 'fmsub.d': 198, 'fmsub.s': 199, 'fmul.d': 200, 'fmul.s': 201, 'fmv.d.x': 202, 'fmv.w.x': 203, 'fmv.x.d': 204, 'fmv.x.w': 205, 'fnmadd.d': 206, 'fnmadd.s': 207, 'fnmsub.d': 208, 'fnmsub.
s': 209, 'fsd': 210, 'fsgnj.d': 211, 'fsgnj.s': 212, 'fsgnjn.d': 213, 'fsgnjn.s': 214, 'fsgnjx.d': 215, 'fsgnjx.s': 216, 'fsqrt.d': 217, 'fsqrt.s': 218, 'fsub.d': 219, 'fsub.s': 220, 'fsw': 221, 'csrrc': 222, 'csrrci': 223, 'csrrs': 224, 'csrrsi': 225, 'csrrw': 226, 'csrrwi': 227, 'fence.i': 228}
        """

    def separate_disp_and_reg(self, s):

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

        pattern = re.compile(r'^-?\d+(\.\d+)?$')
        return bool(pattern.match(s))

    def is_address_string(self, s):

        pattern = re.compile(r'^0x[0-9a-fA-F]+$')
        return bool(pattern.match(s))

    def zero_reg(self, instr):

        return [instr[0], '<E>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']

    def one_reg_num(self, instr):

        pattern = r'^-?\d+$'
        if re.match(pattern, instr[1]):  # "j, -1666"
            return [instr[0], '<CONST>', '<E>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
        else:
            return [instr[0], instr[1], '<E>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']

    def two_reg(self, instr):

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

        return [instr[0], '<D>', instr[1], '<S>', instr[2], instr[3], instr[4], '<E>']

    def tokenize_instruction(self, instruction: str) -> List[str]:
        """
        transform RISC-V instruction into token list,
        eg: "add a5,s1,a0" -> ['add', '<D>', 'a5', '<S>', 's1', 'a0', '<E>', '<PAD>']

        Args:
            instruction: "add a5,s1,a0"

        Returns:
            ['add', '<D>', 'a5', '<S>', 's1', 'a0', '<E>', '<PAD>']
        """
        pattern = r'[^ ,\t]+'  # match non-space characters
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
        Encoded the instruction into token ID sequence,
        eg: "add a5,s1,a0" -> [ 72, 5, 24, 6, 18, 19, 4, 0]

        Args:
            instruction: "add a5,s1,a0"

        Returns:
            [ 72, 5, 24, 6, 18, 19, 4, 0]
        """
        tokens = self.tokenize_instruction(instruction)
        encoded = [self.vocab.get(token, self.vocab.get('<PAD>', 0)) for token in tokens]

        return encoded

    def tokenized_bb(self, basic_block: str) -> List[List[str]]:
        """
        Tokenizes a basic block into a list of token lists.
        Args:
            basic_block: A string representing the basic block, with instructions separated by newline characters.

        Returns:
            A list of tokenized lists.
        """
        bb_tokens = []
        pattern = r'[^ ,\t]+'
        bb = [re.findall(pattern, s) for s in basic_block.strip().split("\\n")]

        for instruction in bb:
            tokenized = self.tokenize_instruction(instruction)
            bb_tokens.append(tokenized)

        return bb_tokens

# if __name__ == "__main__":
#     tokenizer = RISCVTokenizer()
#     tokenizer._build_vocab_manually()