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

frrrr_insts = {n: FRRRRFmtInst(n, 'f0', 'f0', 'f0', 'f0')
              for n in ["fmadd.d", "fmadd.s", "fmsub.d", "fmsub.s", "fnmadd.d", "fnmadd.s", "fnmsub.d", "fnmsub.s"]} # 8

frr_insts = {n: FRRFmtInst(n, 'f0', 'f0')
             for n in ["fclass.d", "fclass.s", "fcvt.d.l", "fcvt.d.lu", "fcvt.d.s", "fcvt.d.w", "fcvt.d.wu",
                        "fcvt.l.d", "fcvt.l.s", "fcvt.lu.d", "fcvt.lu.s", "fcvt.s.d", "fcvt.s.l", "fcvt.s.lu", "fcvt.s.w", "fcvt.s.wu",
                        "fcvt.w.d", "fcvt.w.s", "fcvt.wu.d", "fcvt.wu.s", "fmv.d.x", "fmv.w.x", "fmv.x.d", "fmv.x.w", "fsqrt.d", "fsqrt.s"]} # 26

frrr_insts = {n:RFmtInst(n, 'f0', 'f0', 'f0')
               for n in ['fadd.d', 'fadd.s', 'fdiv.d', 'fdiv.s', "fmax.d", "fmax.s", "fmin.d", "fmin.s", "fmul.d", "fmul.s",
                         "fsgnj.d", "fsgnj.s", "fsgnjn.d", "fsgnjn.s", "fsgnjx.d", "fsgnjx.s", "fsub.d", "fsub.s",
                         'feq.d', 'feq.s', 'fle.d', 'fle.s', 'flt.d', 'flt.s']} # 24

frri_insts = {n:IFmtInst(n, 'f0', 'x0', 0)
              for n in ['fld', 'flw', 'fsw', 'fsd']} # 4


shifts_arithmetic_logical_insts = {
    **{n: RFmtInst(n, 'x0', 'x0', 'x0')
     for n in ['add', 'addw', 'and', 'sll', 'sllw', 'sra',
               'sraw', 'srl', 'srlw', 'sub', 'subw', 'xor', 'or']},

    **{n: IFmtInst(n, 'x0', 'x0', 0)
     for n in ['addi', 'addiw', 'andi', 'ori', 'slli', 'slliw',
               'srai', 'sraiw', 'srli', 'srliw', 'xori']},

    **ufmt_insts
} # 26

compare_insts = {
    **{n: RFmtInst(n, 'x0', 'x0', 'x0')
     for n in ['slt', 'sltu']},

    **{n: IFmtInst(n, 'x0', 'x0', 0)
     for n in ['slti', 'sltiu']}
} # 4

mul_div_insts = {
    n: RFmtInst(n, 'x0', 'x0', 'x0')
    for n in ['div', 'divu', 'divuw', 'divw', 'mul', 'mulh', 'mulhsu',
              'mulhu', 'mulw', 'rem', 'remu', 'remuw', 'remw']
} # 13

load_insts = {
    n: LoadInst(n, 'x0', 'x0', 0)
    for n in ['lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu']
} # 7

store_insts = {
    n: StoreInst(n, 'x0', 'x0', 0)
    for n in ['sb', 'sd', 'sh', 'sw']
} # 4

branch_insts = {
    n: BFmtInst(n, 'x0', 'x0', 0)
    for n in ['beq', 'bge', 'bgeu', 'blt', 'bltu', 'bne']
}

jalr_inst = JalrInst('jalr', 'x0', 'x0', 0)
jal_inst = JFmtInst('jal', 'x0', 0)

normal_insts = {**rfmt_insts, **ifmt_insts, **load_insts, **store_insts}
exit_insts = {**branch_insts, 'jalr': jalr_inst, 'jal': jal_inst}  # 8

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


class DependencyAnalyzer:
    """
    Data dependency analyzer for RISC-V assembly basic blocks.

    Analyzes three types of data dependencies:
    - RAW (Read After Write): Occurs when an instruction reads a register that was written by a previous instruction
    - WAR (Write After Read): Occurs when an instruction writes to a register that was read by a previous instruction
    - WAW (Write After Write): Occurs when an instruction writes to a register that was written by a previous instruction
    """

    def __init__(self):
        # Initialize dependency counters
        self.raw_count = 0  # Read After Write
        self.war_count = 0  # Write After Read
        self.waw_count = 0  # Write After Write

        # Lists for detailed reporting
        self.raw_deps = []  # List storing (def_idx, use_idx, reg) tuples
        self.war_deps = []  # List storing (use_idx, def_idx, reg) tuples
        self.waw_deps = []  # List storing (old_def_idx, new_def_idx, reg) tuples

    def analyze(self, basic_block):
        """
        Analyzes data dependencies in a RISC-V assembly basic block.

        Parameters:
            basic_block: List of instruction objects, each should provide get_def() and get_uses() methods

        Returns:
            Tuple containing counts of the three dependency types (raw_count, war_count, waw_count)
        """
        # Reset counters and dependency lists
        self.raw_count = 0
        self.war_count = 0
        self.waw_count = 0
        self.raw_deps = []
        self.war_deps = []
        self.waw_deps = []

        # Initialize tracking dictionaries
        last_def = {}  # register -> index of instruction that last defined it
        readers = {}  # register -> set of instruction indices that read it after its last definition

        # Analyze each instruction
        for i, insn in enumerate(basic_block):
            def_reg = insn.get_def()  # Destination register (written)
            use_regs = insn.get_uses()  # Source registers (read)

            # Check for RAW dependencies (current instruction reads a register written by a previous instruction)
            for reg in use_regs:
                if reg in last_def:
                    self.raw_count += 1
                    self.raw_deps.append((last_def[reg], i, reg))

            # Check for WAR dependencies (current instruction writes to a register read by a previous instruction)
            if def_reg and def_reg in readers:
                for reader_idx in readers[def_reg]:
                    self.war_count += 1
                    self.war_deps.append((reader_idx, i, def_reg))

            # Check for WAW dependencies (current instruction writes to a register written by a previous instruction)
            if def_reg and def_reg in last_def:
                self.waw_count += 1
                self.waw_deps.append((last_def[def_reg], i, def_reg))

            # Update tracking information
            # Add current instruction to readers for all registers it reads
            for reg in use_regs:
                if reg not in readers:
                    readers[reg] = set()
                readers[reg].add(i)

            # Update last definition for written register
            if def_reg:
                last_def[def_reg] = i
                # Clear the readers set for this register as it has a new definition
                readers[def_reg] = set()

        return self.raw_count, self.war_count, self.waw_count

    def print_summary(self):
        """
        Print a summary of detected dependencies.
        """
        print(f"Total dependencies: {self.raw_count + self.war_count + self.waw_count}")
        print(f"RAW dependencies: {self.raw_count}")
        print(f"WAR dependencies: {self.war_count}")
        print(f"WAW dependencies: {self.waw_count}")

    def print_details(self):
        """
        Print detailed information about detected dependencies.
        """
        print("RAW dependencies:")
        for def_idx, use_idx, reg in self.raw_deps:
            print(f"  Instruction {use_idx} reads {reg}, which was written by instruction {def_idx}")

        print("WAR dependencies:")
        for use_idx, def_idx, reg in self.war_deps:
            print(f"  Instruction {def_idx} writes to {reg}, which was read by instruction {use_idx}")

        print("WAW dependencies:")
        for old_def_idx, new_def_idx, reg in self.waw_deps:
            print(f"  Instruction {new_def_idx} writes to {reg}, which was also written by instruction {old_def_idx}")



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


def gen_block_vector(num_insts: int = 100, ratios: list = [0.5, 0.2, 0.1, 0.1, 0.1], dependency_flags=[1, 1, 1], seed: int = None, depth=3):
    if seed is not None:
        random.seed(seed)

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

    # first stage：allocate registers randomly
    for inst in block:
        if inst.get_def():
            reg = gen_reg()
            inst.set_def(reg)

        num_uses = len(inst.get_uses())
        if num_uses > 0:
            inst.set_uses([gen_reg() for _ in range(num_uses)])

    # second stage: inject dependencies as needed
    # Create WAW dependency
    if dependency_flags[0]:
        writer_idx = 0
        target_idx = depth
        if target_idx < len(block):
            # Get a register for the WAW dependency
            reg = block[writer_idx].get_def()
            print(reg)
            block[target_idx].set_def(reg)

    # Create RAW dependency
    if dependency_flags[1]:
        writer_idx = 1
        reader_idx = depth
        if reader_idx < len(block):
            reg = block[writer_idx].get_def()
            uses = block[reader_idx].get_uses()
            if len(uses) > 0:
                new_uses = uses.copy()
                new_uses[0] = reg
                block[reader_idx].set_uses(new_uses)

    # Create WAR dependency
    if dependency_flags[2]:
        reader_idx = 2
        writer_idx = depth
        if writer_idx < len(block):
            uses = block[reader_idx].get_uses()
            if len(uses) > 0:
                block[writer_idx].set_def(uses[0])

    # third stage: assign immediates
    for inst in block:
        if hasattr(inst, 'imm'):
            if inst.name in ["srli", "srai", "slli"]:
                inst.imm = gen_imm(0, 63, 1)
            elif inst.name in ["srliw", "slliw", "sraiw"]:
                inst.imm = gen_imm(0, 31, 1)
            else:
                inst.imm = gen_imm(-8, 8, 8)

    return block
