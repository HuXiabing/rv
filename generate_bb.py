import pprint as pp
import sys
import json
from rvmca.gen import gen_block
from rvmca.gen.inst_gen import gen_block_vector, DependencyAnalyzer
import argparse
import os

def test2(len_bb):
    block = gen_block(len_bb)
    print(block)
    analyzer = DependencyAnalyzer()
    raw, war, waw = analyzer.analyze(block)
    print(f"Analysis results: RAW={raw}, WAR={war}, WAW={waw}")
    # analyzer.print_summary()
    # print()
    # analyzer.print_details()

    return block


if __name__ == "__main__":
    file_path = "experiments/transformer_v1_20250312_223609/analysis_epoch_8/analysis_summary.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs('./random_generate/asm', exist_ok=True)

    num_bb = 2000
    num = 0
    for key, value in data["block_dict"].items():
        cnt = round(value * num_bb)
        # 生成指定长度基本块的个数
        for i in range(cnt):
            block = gen_block_vector(normalized_vector = data["instruction_vec"], len_bb = int(key))

            with open(f'./random_generate/asm/test{num}_nojump.S', 'w') as file:
                # file.write("# LLVM-MCA-BEGIN A simple example" + '\n')
                for line in block:
                    file.write(line.code + '\n')
                # file.write("# LLVM-MCA-END")
            num += 1
