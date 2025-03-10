# MIT License
#
# Copyright (c) 2024 Xuezheng (xuezhengxu@126.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Instruction Generator"""

import random
from secrets import choice
from typing import Dict
from rvmca.prog.inst import *
from rvmca.trans.block import *
from rvmca.prog.reg import XREG, FREG

rfmt_insts = {
    n: RFmtInst(n, 'x0', 'x0', 'x0')
    for n in ['add', 'addw', 'and', 'or'
              'div', 'divu', 'divuw', 'divw','mul', 'mulh', 'mulhsu', 'mulhu', 'mulw', 'rem', 'remu','remuw', 'remw',  # M
              'sll', 'sllw', 'slt', 'sltu', 'sra',
              'sraw', 'srl', 'srlw', 'sub', 'subw', 'xor']
}

ifmt_insts = {
    n: IFmtInst(n, 'x0', 'x0', 0)
    for n in ['addi', 'addiw', 'andi', 'ori', 'slli', 'slliw',
              'slti', 'sltiu', 'srai', 'sraiw', 'srli', 'srliw', 'xori']
}


ufmt_insts = {
    n: UFmtInst(n, 'x0', 0)
    for n in ['auipc', 'lui']
}

frrr_insts = {n: FRRRRFmtInst(n, 'f0', 'f0', 'f0', 'f0')
              for n in ['']}

shifts_arithmetic_logical_insts = {
    **{n: RFmtInst(n, 'x0', 'x0', 'x0')
     for n in ['add', 'addw', 'and', 'sll', 'sllw', 'sra',
               'sraw', 'srl', 'srlw', 'sub', 'subw', 'xor']},

    **{n: IFmtInst(n, 'x0', 'x0', 0)
     for n in ['addi', 'addiw', 'andi', 'ori', 'slli', 'slliw',
               'srai', 'sraiw', 'srli', 'srliw', 'xori']},

    **ufmt_insts
}

compare_insts = {
    **{n: RFmtInst(n, 'x0', 'x0', 'x0')
     for n in ['slt', 'sltu']},

    **{n: IFmtInst(n, 'x0', 'x0', 0)
     for n in ['slti', 'sltiu']}
}

mul_div_insts = {
    n: RFmtInst(n, 'x0', 'x0', 'x0')
    for n in ['div', 'divu', 'divuw', 'divw', 'mul', 'mulh', 'mulhsu',
              'mulhu', 'mulw', 'rem', 'remu', 'remuw', 'remw']
}

load_insts = {
    n: LoadInst(n, 'x0', 'x0', 0)
    for n in ['lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu']
}

store_insts = {
    n: StoreInst(n, 'x0', 'x0', 0)
    for n in ['sb', 'sd', 'sh', 'sw']
}

branch_insts = {
    n: BFmtInst(n, 'x0', 'x0', 0)
    for n in ['beq', 'bge', 'bgeu', 'blt', 'bltu', 'bne']
}

jalr_inst = JalrInst('jalr', 'x0', 'x0', 0)
jal_inst = JFmtInst('jal', 'x0', 0)

normal_insts = {**rfmt_insts, **ifmt_insts, **load_insts, **store_insts}
exit_insts = {**branch_insts, 'jalr': jalr_inst, 'jal': jal_inst}

riscv_insts = {**normal_insts, **exit_insts}
##############################################################################




##############################################################################

def gen_reg(candidates: List = None):
    if candidates is None:
        candidates = XREG
    return random.choice(candidates)

def gen_inst(candidates: Dict = None):
    if candidates is None:
        candidates = riscv_insts
    return copy.deepcopy(random.choice(list(candidates.values())))
    # choice = random.choice(list(candidates.keys()))
    # return candidates[choice]

def gen_imm(start=-8, end=8, divisor=1):
    return random.randint(start // divisor, end // divisor) * divisor

def dependency_analyzer(block, depth=1):

    waw = [0] * 10
    raw = [0] * 10
    war = [0] * 10

    for i, insn in enumerate(block):
        def_reg_current = insn.get_def()
        uses_regs_current = insn.get_uses()

        for j, insnj in enumerate(block):
            if j > i:
                def_reg_next = insnj.get_def()
                uses_regs_next = insnj.get_uses()

                if def_reg_current and def_reg_current == def_reg_next:
                    # print(insn)
                    # print(insnj)
                    # print(def_reg_current)
                    # print(j, j-i)
                    waw[j-i] += 1
                    # print(j)

                if def_reg_current in uses_regs_next:
                    raw[j-i] += 1

                for uses_reg in uses_regs_current:
                    if uses_reg == def_reg_next:
                        war[j-i] += 1


    return raw, war, waw

def gen_block(num_insts: int = 10, seed: int = None):
    if seed is not None:
        random.seed(seed)

    # generate dummy instructions for a block
    block = [gen_inst(normal_insts) for _ in range(num_insts)]
    # exit_inst = legalize_jump_or_branch_inst(gen_inst(exit_insts))
    # block.append(exit_inst)

    # register allocation
    for inst in block:
        if inst.get_def():
            reg = gen_reg()
            inst.set_def(reg)

        num_uses = len(inst.get_uses())
        if num_uses > 0:
            inst.set_uses([gen_reg() for _ in range(num_uses)])

    # # assign immediate
    # for inst in block[:-1]:
    #     if hasattr(inst, 'imm'):
    #         inst.imm = gen_imm(-8, 8, 8)
        # assign immediate
    for inst in block[:-1]:
        if hasattr(inst, 'imm'):
            if inst.name in ["srli", "srai", "slli"]:
                inst.imm = gen_imm(0, 63, 1)
            elif inst.name in ["srliw", "slliw", "sraiw"]:
                inst.imm = gen_imm(0, 31, 1)
            else:
                inst.imm = gen_imm(-8, 8, 8)

    return block

# def gen_block_vector(vec: list = [10, 0.5, 0.2, 0.1, 0.1, 0.1, 1, 1, 1], seed: int = None, depth = 1): #waw 1,war 2,  raw3
#
#     # [shifts_arithmetic_logical_ratio, compare_ratio, mul_div_ratio, load_ratio, store_ratio]
#
#     if seed is not None:
#         random.seed(seed)
#
#     # generate dummy instructions for a block
#
#     num_insts = vec[0]
#     print("生成的基本块指令数为",num_insts)
#     total_ratio = sum(vec[1:5])
#     shifts_arithmetic_logical_insts_num = round(num_insts * vec[1] / total_ratio)
#     compare_insts_num = round(num_insts * vec[2] / total_ratio)
#     mul_div_insts_num = round(num_insts * vec[3] / total_ratio)
#     load_insts_num = round(num_insts * vec[4] / total_ratio)
#     store_insts_num = round(num_insts * vec[5] / total_ratio)
#
#     block = []
#
#     for i in range(shifts_arithmetic_logical_insts_num):
#         block.append(gen_inst(shifts_arithmetic_logical_insts))
#
#     for i in range(compare_insts_num):
#         block.append(gen_inst(compare_insts))
#
#     for i in range(mul_div_insts_num):
#         block.append(gen_inst(mul_div_insts))
#
#     for i in range(load_insts_num):
#         block.append(gen_inst(load_insts))
#
#     for i in range(store_insts_num):
#         block.append(gen_inst(store_insts))
#
#     random.shuffle(block)
#     registers = XREG.copy()
#
#     # exist_war = vec[6] exist_raw = vec[7] exist_waw = vec[8]
#
#     # register allocation
#     if vec[6]:
#         reg = registers.pop()
#         num_uses = len(block[0].get_uses())
#         block[0].set_uses([reg] + [registers.pop() for _ in range(1, num_uses)])
#         block[0].set_def(registers.pop())
#         block[depth].set_def(reg)
#         num_uses = len(block[depth].get_uses())
#         if num_uses > 0:
#             block[depth].set_uses([gen_reg() for _ in range(num_uses)])
#
#
#     if vec[7]:
#         reg = registers.pop()
#         block[2].set_def(reg)
#         num_uses = len(block[depth+2].get_uses())
#         block[depth+2].set_uses([reg] + [registers.pop() for _ in range(1, num_uses)])
#         block[depth+2].set_def(registers.pop())
#         if len(block[2].get_uses()) > 0:
#             block[2].set_uses([gen_reg() for _ in range(len(block[2].get_uses()))])
#
#     if vec[8]:
#         reg = registers.pop()
#         block[1].set_def(reg)
#         if len(block[1].get_uses()) > 0:
#             block[1].set_uses([registers.pop() for _ in range(len(block[1].get_uses()))])
#         block[depth+1].set_def(reg)
#         if len(block[depth+1].get_uses()) > 0:
#             block[depth+1].set_uses([registers.pop() for _ in range(len(block[depth+1].get_uses()))])
#
#     for inst in block:
#         if inst.get_def() == Reg(0):
#             if inst.get_def():
#                 reg = gen_reg()
#                 inst.set_def(reg)
#             num_uses = len(inst.get_uses())
#             if num_uses > 0:
#                 inst.set_uses([gen_reg() for _ in range(num_uses)])
#
#     # assign immediate
#     for inst in block[:-1]:
#         if hasattr(inst, 'imm'):
#             if inst.name in ["srli", "srai", "slli"]:
#                 inst.imm = gen_imm(0,63,1)
#             elif inst.name in ["srliw", "slliw", "sraiw"]:
#                 inst.imm = gen_imm(0,31,1)
#             else:
#                 inst.imm = gen_imm(-8, 8, 8)
#
#     return block

def gen_block_vector(vec: list = [10, 0.5, 0.2, 0.1, 0.1, 0.1,], dependency_flags=[1, 1, 1], seed: int = None, depth=1):
    if seed is not None:
        random.seed(seed)

    num_insts = vec[0]
    ratios = vec[1:6]

    # 动态生成策略
    BASE_SIZE = 150
    if num_insts < BASE_SIZE:
        # 按比例生成基础样本池
        pool = []
        for inst_type, ratio in zip([shifts_arithmetic_logical_insts,
                                     compare_insts,
                                     mul_div_insts,
                                     load_insts,
                                     store_insts], ratios):
            n = int(BASE_SIZE * ratio)
            pool += [gen_inst(inst_type) for _ in range(n)]

        random.shuffle(pool)
        block = pool[:num_insts]
    else:
        # 直接按比例生成目标长度
        block = []
        for inst_type, ratio in zip([shifts_arithmetic_logical_insts,
                                     compare_insts,
                                     mul_div_insts,
                                     load_insts,
                                     store_insts], ratios):
            n = int(num_insts * ratio)
            block += [gen_inst(inst_type) for _ in range(n)]

        while len(block) < num_insts:
            block.append(gen_inst(shifts_arithmetic_logical_insts))
        block = block[:num_insts]
        random.shuffle(block)

    # 第一阶段：完全随机分配寄存器

    for inst in block:
        if inst.get_def():
            reg = gen_reg()
            inst.set_def(reg)

        num_uses = len(inst.get_uses())
        if num_uses > 0:
            inst.set_uses([gen_reg() for _ in range(num_uses)])

    registers = XREG.copy()
    if dependency_flags[0]:
        target_idx = depth
        if target_idx < len(block):
            reg = registers.pop()
            num_uses = len(block[0].get_uses())
            block[0].set_uses([reg] + [registers.pop() for _ in range(1, num_uses)])
            block[0].set_def(registers.pop())
            block[target_idx].set_def(reg)
            num_uses = len(block[target_idx].get_uses())
            if num_uses > 0:
                block[target_idx].set_uses([gen_reg() for _ in range(num_uses)])

    if dependency_flags[1]:
        writer_idx = 2
        reader_idx = depth + 2
        if reader_idx < len(block):
            reg = registers.pop()
            block[writer_idx].set_def(reg)
            num_uses = len(block[reader_idx].get_uses())
            block[reader_idx].set_uses([reg] + [registers.pop() for _ in range(1, num_uses)])
            block[reader_idx].set_def(registers.pop())
            if len(block[writer_idx].get_uses()) > 0:
                block[writer_idx].set_uses([gen_reg() for _ in range(len(block[writer_idx].get_uses()))])

    if dependency_flags[2]:
        reader_idx = 1
        writer_idx = depth + 1
        if writer_idx < len(block):
            reg = registers.pop()
            block[reader_idx].set_def(reg)
            if len(block[reader_idx].get_uses()) > 0:
                block[reader_idx].set_uses([registers.pop() for _ in range(len(block[reader_idx].get_uses()))])
            block[writer_idx].set_def(reg)
            if len(block[writer_idx].get_uses()) > 0:
                block[writer_idx].set_uses([registers.pop() for _ in range(len(block[writer_idx].get_uses()))])

    # 第二阶段：按需注入依赖

    # # WAW依赖
    # if dependency_flags[0]:
    #     target_idx = depth
    #     if target_idx < len(block):
    #         reg = block[0].get_def()
    #         block[target_idx].set_def(reg)
    #
    #
    # # WAR依赖
    # if dependency_flags[1]:
    #     writer_idx = 1
    #     reader_idx = depth + 1
    #     if reader_idx < len(block):
    #         reg = block[writer_idx].get_def()
    #         if block[reader_idx].get_uses():
    #             block[reader_idx].set_uses([reg] + block[reader_idx].get_uses()[1:])
    #
    # # RAW依赖
    # if dependency_flags[2]:
    #     reader_idx = 2
    #     writer_idx = depth + 2
    #     if writer_idx < len(block):
    #         reg = block[writer_idx].get_def()
    #         if block[reader_idx].get_uses():
    #             block[reader_idx].set_uses([reg] + block[reader_idx].get_uses()[1:])

    # 第三阶段：处理立即数
    for inst in block:
        if hasattr(inst, 'imm'):
            if inst.name in ["srli", "srai", "slli"]:
                inst.imm = gen_imm(0, 63, 1)
            elif inst.name in ["srliw", "slliw", "sraiw"]:
                inst.imm = gen_imm(0, 31, 1)
            else:
                inst.imm = gen_imm(-8, 8, 8)

    return block


# def gen_block_vector(vec: list = [10, 0.5, 0.2, 0.1, 0.1, 0.1, 1, 1, 1], seed: int = None, depth=1):
#     """
#     Generate a block of instructions with specified ratios and dependencies.
#
#     Args:
#         vec: A list representing [num_insts, shifts_arithmetic_logical_ratio, compare_ratio,
#                                  mul_div_ratio, load_ratio, store_ratio,
#                                  exist_waw, exist_war, exist_raw]
#         seed: Random seed for reproducibility
#         depth: Depth of dependencies to create
#
#     Returns:
#         A list of instruction objects
#     """
#     if seed is not None:
#         random.seed(seed)
#
#     # Extract parameters from vector
#     num_insts = int(vec[0])
#     exist_waw = bool(vec[6])
#     exist_war = bool(vec[7])
#     exist_raw = bool(vec[8])
#
#     print(f"Generating basic block with {num_insts} instructions")
#
#     # Calculate the number of instructions to generate initially
#     initial_pool_size = max(150, num_insts)
#
#     # Calculate instruction type counts based on ratios
#     total_ratio = sum(vec[1:6])
#     if total_ratio == 0:
#         # Fallback to equal distribution if all ratios are 0
#         vec[1:6] = [0.2, 0.2, 0.2, 0.2, 0.2]
#         total_ratio = 1.0
#
#     type_counts = [
#         round(initial_pool_size * vec[1] / total_ratio),  # shifts_arithmetic_logical
#         round(initial_pool_size * vec[2] / total_ratio),  # compare
#         round(initial_pool_size * vec[3] / total_ratio),  # mul_div
#         round(initial_pool_size * vec[4] / total_ratio),  # load
#         round(initial_pool_size * vec[5] / total_ratio)  # store
#     ]
#
#     # Generate instruction pool
#     instruction_pool = []
#     for count, inst_type in zip(type_counts, [
#         shifts_arithmetic_logical_insts,
#         compare_insts,
#         mul_div_insts,
#         load_insts,
#         store_insts
#     ]):
#         for _ in range(count):
#             instruction_pool.append(gen_inst(inst_type))
#
#     # Shuffle the instruction pool
#     random.shuffle(instruction_pool)
#
#     # Select the required number of instructions
#     if num_insts < len(instruction_pool):
#         block = random.sample(instruction_pool, num_insts)
#     else:
#         block = instruction_pool
#
#     # Random register allocation for all instructions first
#     for inst in block:
#         if inst.get_def():
#             reg = gen_reg()
#             inst.set_def(reg)
#
#         num_uses = len(inst.get_uses())
#         if num_uses > 0:
#             inst.set_uses([gen_reg() for _ in range(num_uses)])
#
#     # Create dependencies if needed and if possible
#     if exist_waw or exist_war or exist_raw:
#         # Check if dependencies can be inserted based on depth and block size
#         can_insert_dependencies = len(block) > depth + 2  # Need at least depth+3 instructions
#
#         if can_insert_dependencies:
#             available_regs = XREG.copy()
#             random.shuffle(available_regs)  # Shuffle to add more randomness
#
#             # WAW dependency: same register defined at positions 1 and depth+1
#             if exist_waw:
#                 reg = available_regs.pop() if available_regs else gen_reg()
#                 idx1 = 1
#                 idx2 = depth + 1
#
#                 if idx1 < len(block) and idx2 < len(block):
#                     block[idx1].set_def(reg)
#                     block[idx2].set_def(reg)
#
#             # WAR dependency: register used at position 0, defined at position depth
#             if exist_war:
#                 reg = available_regs.pop() if available_regs else gen_reg()
#                 idx1 = 0
#                 idx2 = depth
#
#                 if idx1 < len(block) and idx2 < len(block) and len(block[idx1].get_uses()) > 0:
#                     uses = block[idx1].get_uses()
#                     block[idx1].set_uses([reg] + [gen_reg() for _ in range(len(uses) - 1)] if len(uses) > 1 else [reg])
#                     block[idx2].set_def(reg)
#
#             # RAW dependency: register defined at position 2, used at position depth+2
#             if exist_raw:
#                 reg = available_regs.pop() if available_regs else gen_reg()
#                 idx1 = 2
#                 idx2 = depth + 2
#
#                 if idx1 < len(block) and idx2 < len(block) and len(block[idx2].get_uses()) > 0:
#                     block[idx1].set_def(reg)
#                     uses = block[idx2].get_uses()
#                     block[idx2].set_uses([reg] + [gen_reg() for _ in range(len(uses) - 1)] if len(uses) > 1 else [reg])
#         else:
#             print("Warning: Block size too small for requested dependencies. Dependencies not inserted.")
#
#     # Assign immediates
#     for inst in block:
#         if hasattr(inst, 'imm'):
#             if inst.name in ["srli", "srai", "slli"]:
#                 inst.imm = gen_imm(0, 63, 1)
#             elif inst.name in ["srliw", "slliw", "sraiw"]:
#                 inst.imm = gen_imm(0, 31, 1)
#             else:
#                 inst.imm = gen_imm(-8, 8, 8)
#
#     return block