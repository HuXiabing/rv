#!/usr/bin/env python
"""
增量数据预处理脚本 - 将新的原始JSON数据转换为处理后的JSON格式和HDF5格式，并合并到已有数据中
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import get_config
from data.tokenizer import RISCVTokenizer
from utils import set_seed


def merge_json_files(existing_json_path, new_json_path, output_json_path):
    """
    合并两个JSON文件
    """
    # 读取已有JSON数据
    with open(existing_json_path, 'r') as f:
        existing_data = json.load(f)

    # 读取新的JSON数据
    with open(new_json_path, 'r') as f:
        new_data = json.load(f)

    # 合并数据
    merged_data = existing_data + new_data

    # 保存合并后的JSON数据
    with open(output_json_path, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"合并后的JSON数据已保存到: {output_json_path}")


def merge_h5_files(existing_h5_path, new_h5_path, output_h5_path):
    """
    合并两个HDF5文件
    """
    # 读取已有HDF5文件
    with h5py.File(existing_h5_path, 'r') as f1:
        X1 = f1['X'][:]
        instruction_counts1 = f1['instruction_counts'][:]
        Y1 = f1['Y'][:]
        has_instruction_text = 'instruction_text' in f1
        if has_instruction_text:
            instruction_text1 = f1['instruction_text'][:]

        # 获取属性
        num_samples1 = f1.attrs['num_samples']
        max_instr_count = f1.attrs.get('max_instr_count', 20)
        max_instr_length = f1.attrs.get('max_instr_length', 8)

    # 读取新的HDF5文件
    with h5py.File(new_h5_path, 'r') as f2:
        X2 = f2['X'][:]
        instruction_counts2 = f2['instruction_counts'][:]
        Y2 = f2['Y'][:]
        if has_instruction_text:
            instruction_text2 = f2['instruction_text'][:]

        num_samples2 = f2.attrs['num_samples']

    # 合并数据
    X_merged = np.vstack([X1, X2])
    instruction_counts_merged = np.concatenate([instruction_counts1, instruction_counts2])
    Y_merged = np.concatenate([Y1, Y2])

    if has_instruction_text:
        instruction_text_merged = np.concatenate([instruction_text1, instruction_text2])

    # 创建新文件
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    with h5py.File(output_h5_path, 'w') as f_out:
        # 创建数据集
        f_out.create_dataset('X', data=X_merged, compression='gzip')
        f_out.create_dataset('instruction_counts', data=instruction_counts_merged)
        f_out.create_dataset('Y', data=Y_merged)

        if has_instruction_text:
            # 创建字符串数据集
            dt = h5py.special_dtype(vlen=str)
            f_out.create_dataset('instruction_text', data=instruction_text_merged, dtype=dt)

        # 设置属性
        f_out.attrs['num_samples'] = num_samples1 + num_samples2
        f_out.attrs['max_instr_count'] = max_instr_count
        f_out.attrs['max_instr_length'] = max_instr_length

    print(f"合并后的HDF5文件已保存到: {output_h5_path}")


def main():
    parser = argparse.ArgumentParser(description="RISC-V指令吞吐量增量数据预处理")

    # 数据参数
    parser.add_argument("--raw_data", type=str, required=True, help="新的原始JSON数据文件路径")
    parser.add_argument("--existing_train_json", type=str, default="data/train_data.json",
                        help="已有的训练JSON数据文件路径")
    parser.add_argument("--existing_train_h5", type=str, default="data/train_data.h5",
                        help="已有的训练HDF5数据文件路径")
    parser.add_argument("--output_dir", type=str, default="data", help="输出目录")

    # 预处理参数
    parser.add_argument("--max_instr_length", type=int, default=8, help="指令最大长度")
    parser.add_argument("--max_instr_count", type=int, default=200, help="样本最大指令数量")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建配置
    config = get_config(
        model_type="transformer",  # 默认使用transformer模型
        max_instr_length=args.max_instr_length,
        max_instr_count=args.max_instr_count,
        raw_data_path=args.raw_data,
        output_dir=args.output_dir,
        seed=args.seed
    )

    # 保存配置
    config_path = os.path.join(args.output_dir, "incremental_preprocess_config.json")
    config.save(config_path)
    print(f"配置已保存到: {config_path}")

    # 创建分词器
    tokenizer = RISCVTokenizer(max_instr_length=args.max_instr_length)

    # 加载新的原始数据
    print(f"正在加载新的原始数据: {args.raw_data}")
    with open(args.raw_data, 'r') as f:
        raw_data = json.load(f)

    # 处理新的数据
    processed_data = []
    for item in tqdm(raw_data, desc="处理数据"):
        instructions = item["instructions"]
        throughput = item["throughput"]

        # 对指令进行分词
        tokenized_instructions = []
        for instr in instructions:
            # 使用自定义分词器处理单条指令
            tokenized = tokenizer.tokenize_instruction(instr)
            tokenized_instructions.append(tokenized)

        # 将分词结果编码为token ID
        encoded_instructions = []
        for tokenized in tokenized_instructions:
            encoded = [tokenizer.vocab.get(token, tokenizer.vocab.get('<PAD>', 0)) for token in tokenized]
            encoded_instructions.append(encoded)

        # 创建处理后的样本
        processed_item = {
            "instructions": instructions,  # 原始指令
            "tokenized": tokenized_instructions,  # 分词后的指令
            "encoded": encoded_instructions,  # 编码后的指令
            "throughput": throughput,  # 吞吐量
            "num_instructions": len(instructions)  # 指令数量
        }

        processed_data.append(processed_item)

    # 保存处理后的JSON数据
    new_json_path = os.path.join(args.output_dir, "new_processed_data.json")
    print(f"保存处理后的JSON数据到: {new_json_path}")
    with open(new_json_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    # 合并到已有的train_data.json
    merged_json_path = os.path.join(args.output_dir, "incremental_train.json")
    merge_json_files(args.existing_train_json, new_json_path, merged_json_path)

    # 创建新的HDF5文件
    new_h5_path = os.path.join(args.output_dir, "new_train_data.h5")
    print("创建新的HDF5文件...")

    def create_hdf5(data, output_path):
        num_samples = len(data)
        X = np.zeros((num_samples, args.max_instr_count, args.max_instr_length), dtype=np.int32)
        instruction_counts = np.zeros(num_samples, dtype=np.int32)
        Y = np.zeros((num_samples,), dtype=np.float32)

        for i, item in enumerate(data):
            # 填充编码的指令
            for j, encoded in enumerate(item["encoded"][:args.max_instr_count]):
                X[i, j, :len(encoded)] = encoded[:args.max_instr_length]

            instruction_counts[i] = min(item["num_instructions"], args.max_instr_count)
            Y[i] = item["throughput"]

        print(f"创建HDF5文件: {output_path}")
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('instruction_counts', data=instruction_counts)
            f.create_dataset('Y', data=Y)

            # 存储原始指令文本
            dt = h5py.special_dtype(vlen=str)
            instr_text = np.array([json.dumps(item["instructions"]) for item in data], dtype=dt)
            f.create_dataset('instruction_text', data=instr_text)

            # 存储元数据
            f.attrs['num_samples'] = num_samples
            f.attrs['max_instr_count'] = args.max_instr_count
            f.attrs['max_instr_length'] = args.max_instr_length

        print(f"已创建HDF5文件: {output_path}")

    create_hdf5(processed_data, new_h5_path)

    # 合并到已有的train_data.h5
    merged_h5_path = os.path.join(args.output_dir, "incremental_train.h5")
    merge_h5_files(args.existing_train_h5, new_h5_path, merged_h5_path)

    print("增量数据预处理完成!")


if __name__ == "__main__":
    main()