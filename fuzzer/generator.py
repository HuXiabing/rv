import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from rvmca.gen.inst_gen import gen_block_vector
import numpy as np
from fuzzer import EnhancedFuzzer
import random
from typing import Dict, List, Tuple
import os
import shutil
import argparse

def rm_all_files(directory: str):
    if os.path.exists(directory):
        # 遍历目录中的所有文件
        for filename in os.listdir(directory):
            # 构建文件的完整路径
            file_path = os.path.join(directory, filename)
            try:
                # 如果是文件，则删除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # 如果是目录，则递归删除（如果需要）
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Directory {directory} does not exist.")

def incre_generator(file_path, val_loss, fuzzer):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fuzzer.update_strategy(
            instruction_avg_loss=data['type_avg_loss'],
            instruction_counts=data['type_counts'],
            block_length_avg_loss=data['block_length_avg_loss'],
            block_length_counts=data['block_length_counts'],
            avg_loss=val_loss
        )

    return fuzzer

def generator(file_path):
    with open(file_path) as f:
        data = json.load(f)

    fuzzer = EnhancedFuzzer(instruction_avg_loss=data['type_avg_loss'],
                            instruction_counts=data['type_counts'],
                            block_length_avg_loss=data['block_length_avg_loss'],
                            block_length_counts=data['block_length_counts'],
                            long_block_penalty=0.1,
                            explore_rate=0.05,
                            temp=0.25,
                            long_block_threshold=128)

    return fuzzer

def generate_blocks(num_blocks, block_length_counts):
    """
    Generate a new dictionary of block lengths based on an existing distribution.

    Args:
        num_blocks: Number of blocks to generate
        block_length_counts: Dictionary mapping length to count

    Returns:
        Dictionary mapping length to generated count
    """
    # Convert all keys to integers
    length_counts = {int(k): v for k, v in block_length_counts.items()}

    # Calculate total count
    total_count = sum(length_counts.values())

    # Create probability distribution
    lengths = list(length_counts.keys())
    probs = [length_counts[l] / total_count for l in lengths]

    # Generate new blocks based on this distribution
    generated_blocks = np.random.choice(lengths, size=num_blocks, p=probs)

    # Count occurrences
    generated_counts = {}
    for length in generated_blocks:
        if length in generated_counts:
            generated_counts[length] += 1
        else:
            generated_counts[length] = 1

    return generated_counts

def riscv_asm_to_hex(assembly_code):
    import subprocess
    import tempfile
    import os
    # 创建临时文件保存汇编代码
    with tempfile.NamedTemporaryFile(suffix='.s', delete=False) as asm_file:
        asm_file.write(assembly_code.encode())
        asm_file_name = asm_file.name

    # 创建临时文件名用于目标文件
    obj_file_name = asm_file_name + '.o'

    try:
        # 使用riscv64-unknown-linux-gnu-as汇编器将汇编代码编译为目标文件
        subprocess.run(['riscv64-unknown-linux-gnu-as', '-march=rv64g', asm_file_name, '-o', obj_file_name], check=True, stderr=subprocess.DEVNULL)

        # 使用riscv64-unknown-linux-gnu-objdump查看目标文件的十六进制内容
        result = subprocess.run(['riscv64-unknown-linux-gnu-objdump', '-d', obj_file_name],
                                capture_output=True, text=True, check=True)

        # 提取十六进制代码
        hex_codes = []
        # print(result.stdout.splitlines())
        for line in result.stdout.splitlines():
            if ':' in line:
                parts = line.split('\t')
                if len(parts) > 1:
                    hex_part = parts[1].strip()
                    if hex_part:
                        hex_codes.append(hex_part)

        return " ".join(hex_codes)

    except subprocess.CalledProcessError as e:
        print(f"Error during compilation: {e}")
        return None
    finally:
        # 清理临时文件
        if os.path.exists(asm_file_name):
            os.remove(asm_file_name)
        if os.path.exists(obj_file_name):
            os.remove(obj_file_name)

if __name__ == "__main__":
    # rm_all_files("./random_generate/asm/")
    # rm_all_files("./random_generate/binary/")

    parser = argparse.ArgumentParser(description="basic block generator")
    parser.add_argument("-n", type=int, default=100, help="number of basic blocks to generate")
    args = parser.parse_args()

    fuzzer = generator('experiments/case_study_20250508_101822/statistics/train_loss_stats_epoch_4.json')
    # fuzzer = generator('../experiments/transformer_20250424_190140/statistics/train_loss_stats_epoch_8.json')
    # fuzzer = incre_generator(
    #     '../experiments/incremental_transformer_20250427_130635/statistics/train_loss_stats_epoch_15.json', 0.073801, fuzzer)
    # fuzzer = incre_generator(
    #     '../experiments/incremental_transformer_20250428_155219/statistics/train_loss_stats_epoch_2.json', 0.073657, fuzzer)
    # fuzzer = incre_generator(
    #     '../experiments/incremental_transformer_20250429_095051/statistics/train_loss_stats_epoch_14.json', 0.062113, fuzzer)
    # fuzzer = incre_generator(
    #     '../experiments/incremental_transformer_20250429_161420/statistics/train_loss_stats_epoch_3.json', 0.061019, fuzzer)
    # fuzzer = incre_generator(
    #     '../experiments/incremental_transformer_20250430_114213/statistics/train_loss_stats_epoch_2.json', 0.058142, fuzzer)
    # fuzzer = incre_generator(
    #     '../experiments/incremental_transformer_20250430_153531/statistics/train_loss_stats_epoch_9.json', 0.055054, fuzzer)
    # fuzzer = incre_generator(
    #     '../experiments/incremental_transformer_20250502_100511/statistics/train_loss_stats_epoch_4.json', 0.054091, fuzzer)
    # fuzzer = incre_generator(
    #     '../experiments/incremental_transformer_20250502_152704/statistics/train_loss_stats_epoch_2.json', 0.054348, fuzzer)

    # incremental_transformer_20250430_114213 - INFO - Experiment completed. Best validation loss: 0.058142 at Epoch 2. Total time: 7491.18 seconds
    # incremental_transformer_20250430_153531 - INFO - Experiment completed. Best validation loss: 0.055054 at Epoch 9. Total time: 16032.83 seconds
    # 2025-05-02 13:13:14,955 - incremental_transformer_20250502_100511 - INFO - Experiment completed. Best validation loss: 0.054091 at Epoch 4. Total time: 11283.70 seconds


    length_plan = fuzzer.plan_generation(args.n)

    print("\n开始生成基本块...")
    oprand_count = {cat: 0 for cat in fuzzer.type_order}
    blocks = []
    cnt = 0
    for length, count in length_plan.items():
        print(length, count)
        if int(length) < 21:
            cnt += count
        for i in range(count):
            block = fuzzer.generate(length)
            blocks.append({"asm": "\\n".join([i.code for i in block])}) # for mca
            # assembly_code = "\n".join([i.code for i in block])
            # blocks.append({"asm": assembly_code,
            #                "binary": riscv_asm_to_hex(assembly_code)}) # for k230
            for instr in [i.code for i in block]:
                type = fuzzer.instr_categories.get(instr.split()[0])
                if type and type in fuzzer.type_order:
                    oprand_count[type] += 1

    print("cnt less than 21", cnt)
    print(oprand_count)

    # with open(f'./random_generate/asm.json', 'w') as file:
    #     json.dump(blocks, file, indent=2) #for mca

    rm_all_files("./random_generate/")
    chunk_size = 800
    total_chunks = (len(blocks) + chunk_size - 1) // chunk_size
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(blocks))  # Ensure we don't go beyond the list length

        # Create filename with chunk number
        filename = f"./random_generate/asm{i}.json"

        # Write chunk to file
        with open(filename, 'w') as f:
            json.dump(blocks[start_idx:end_idx], f, indent=2)

        print(f"Saved chunk {i + 1}/{total_chunks}: items {start_idx} to {end_idx}")


#-----------------------------------------------------------------------------------------------------------------------------
    # with open('experiments/incremental_lstm_20250413_103441/statistics/train_loss_stats_epoch_4.json') as f:
    #     data = json.load(f)
    # new_blocks = generate_blocks(args.n, data['block_length_counts'])
    #
    # blocks = []
    # cnt = 0
    # for length, count in new_blocks.items():
    #     print(length, count)
    #     if int(length) < 21:
    #         cnt += count
    #     for i in range(count):
    #         block = gen_block(length)
    #         blocks.append({"asm": "\\n".join([i.code for i in block])})
    # with open(f'./random_generate/asm.json', 'w') as file:
    #     json.dump(blocks, file, indent=2)
    # print(cnt)


    # for i in range(1000):
    #     block = gen_block(random.randint(2,15))
    #     with open(f'./random_generate/asm/test{i}_{len(block)}_nojump.S', 'w') as file:
    #         # file.write("# LLVM-MCA-BEGIN A simple example" + '\n')
    #         for line in block:
    #             file.write(line.code + '\n')
    #         # file.write("# LLVM-MCA-END")


    # print(block)
    # analyzer = DependencyAnalyzer()
    # raw, war, waw = analyzer.analyze(block)
    # print(f"Analysis results: RAW={raw}, WAR={war}, WAW={waw}")
    # analyzer.print_summary()
    # print()
    # analyzer.print_details()

