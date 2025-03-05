# import h5py
#
# def check_h5_structure(file_path):
#     with h5py.File(file_path, 'r') as f:
#         print(f"File: {file_path}")
#         print("Datasets:")
#         for key in f.keys():
#             print(f"- {key}")
#         print("Attributes:")
#         for key, value in f.attrs.items():
#             print(f"- {key}: {value}")
#
# check_h5_structure("data/train_data.h5")
# check_h5_structure("data/incremental_train.h5")

import pprint as pp
import sys
import json
from rvmca.gen import gen_block
from rvmca.gen.inst_gen import gen_block_vector, dependency_analyzer
import argparse
import os

def test2(len_bb):
    block = gen_block(len_bb)
    # pp.pprint(block)
    return block

def generate(normalized_vector, len_bb):
    vec = [len_bb] + normalized_vector
    exist_war = 1
    exist_raw = 1
    exist_waw = 1

    vec = vec + [exist_war, exist_raw, exist_waw]
    block = gen_block_vector(vec)
    # block = [i.code for i in block]
    # pp.pprint(block)

    # raw, war, waw = dependency_analyzer(block)
    # print(f"RAW: {raw}, WAR: {war}, WAW: {waw}")
    return block

if __name__ == "__main__":
    file_path = "experiments/transformer_v1_20250304_101004/analysis_epoch_9/analysis_summary.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs('./random_generate_bb', exist_ok=True)

    num_bb = 500
    num = 0
    for key, value in data["block_dict"].items():
        cnt = round(value * num_bb)
        # 生成指定长度基本块的个数
        for i in range(cnt):
            block = generate(normalized_vector = data["instruction_vec"], len_bb = int(key))
            # block = test2(len_bb = int(key))
            # pp.pprint(block)
            with open(f'./random_generate_bb/test{num}_nojump.S', 'w') as file:
                # file.write("# LLVM-MCA-BEGIN A simple example" + '\n')
                for line in block:
                    file.write(line.code + '\n')
                # file.write("# LLVM-MCA-END")
            num += 1

    # block = generate(normalized_vector=data["instruction_vec"], len_bb=153)
    # pp.pprint([i.code for i in block])



