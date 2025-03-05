# !/usr/bin/env python
"""
数据预处理脚本 - 将原始JSON数据转换为处理后的JSON格式和HDF5格式
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


def main():
    parser = argparse.ArgumentParser(description="RISC-V指令吞吐量数据预处理")

    # 数据参数
    parser.add_argument("--raw_data", type=str, required=True, help="原始JSON数据文件路径")
    parser.add_argument("--processed_data", type=str, default="data/processed_data.json",
                        help="处理后的JSON数据文件路径")
    parser.add_argument("--train_data", type=str, default="data/train_data.h5", help="训练数据输出路径(HDF5)")
    parser.add_argument("--val_data", type=str, default="data/val_data.h5", help="验证数据输出路径(HDF5)")
    parser.add_argument("--test_data", type=str, default="data/test_data.h5", help="测试数据输出路径(HDF5)")

    # 预处理参数
    parser.add_argument("--max_instr_length", type=int, default=8, help="指令最大长度")
    parser.add_argument("--max_instr_count", type=int, default=200, help="样本最大指令数量")

    # 数据集划分比例参数
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0, help="测试集比例")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data", help="输出目录")
    parser.add_argument("--experiment_name", type=str, default="default", help="实验名称")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 检查数据集划分比例是否合法
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(f"数据集划分比例之和必须为1.0，当前为{total_ratio}")

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
        processed_data_path=args.processed_data,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # 保存配置
    config_path = os.path.join(args.output_dir, "preprocess_config.json")
    config.save(config_path)
    print(f"配置已保存到: {config_path}")

    # 创建分词器
    tokenizer = RISCVTokenizer(max_instr_length=args.max_instr_length)

    # 加载原始数据
    print(f"正在加载原始数据: {args.raw_data}")
    with open(args.raw_data, 'r') as f:
        raw_data = json.load(f)

    # 处理数据
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
    print(f"保存处理后的JSON数据到: {args.processed_data}")
    with open(args.processed_data, 'w') as f:
        json.dump(processed_data, f, indent=2)

    # 创建HDF5文件
    print("创建HDF5文件...")

    # 划分训练集、验证集和测试集
    import random
    random.seed(args.seed)
    random.shuffle(processed_data)

    total_count = len(processed_data)
    val_count = int(total_count * args.val_ratio)
    train_count = int(total_count * args.train_ratio)
    test_count = total_count - val_count - train_count  # 确保数据集总数正确

    # 按照验证集、训练集、测试集的顺序划分
    val_data = processed_data[:val_count]
    train_data = processed_data[val_count:val_count + train_count]
    test_data = processed_data[val_count + train_count:]

    print(f"数据划分: 验证集 {len(val_data)}条, 训练集 {len(train_data)}条, 测试集 {len(test_data)}条")

    # 保存划分的JSON数据
    train_json_path = os.path.join(args.output_dir, "train_data.json")
    val_json_path = os.path.join(args.output_dir, "val_data.json")
    test_json_path = os.path.join(args.output_dir, "test_data.json")

    with open(train_json_path, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(val_json_path, 'w') as f:
        json.dump(val_data, f, indent=2)

    with open(test_json_path, 'w') as f:
        json.dump(test_data, f, indent=2)

    def create_hdf5(data, output_path):
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:  # 如果路径中包含目录
            os.makedirs(output_dir, exist_ok=True)

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

    # 按照验证集、训练集、测试集的顺序创建HDF5文件
    create_hdf5(val_data, args.val_data)
    create_hdf5(train_data, args.train_data)
    create_hdf5(test_data, args.test_data)

    print("数据预处理完成!")


if __name__ == "__main__":
    main()