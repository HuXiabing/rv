# #!/usr/bin/env python
# """
# 数据预处理脚本 - 将原始JSON数据转换为HDF5格式
# """
#
# import os
# import sys
# import argparse
# import json
# from pathlib import Path
#
# # 添加项目根目录到Python路径
# sys.path.append(str(Path(__file__).resolve().parent.parent))
#
# from config import get_config
# from data import RISCVDataProcessor
# from utils import set_seed
#
#
# def main():
#     parser = argparse.ArgumentParser(description="RISC-V指令吞吐量数据预处理")
#
#     # 数据参数
#     parser.add_argument("--raw_data", type=str, required=True, help="原始JSON数据文件路径")
#     parser.add_argument("--processed_data", type=str, default="data/processed_data.json", help="处理后的JSON数据文件路径")
#     parser.add_argument("--train_data", type=str, default="data/train_data.h5", help="训练数据输出路径(HDF5)")
#     parser.add_argument("--val_data", type=str, default="data/val_data.h5", help="验证数据输出路径(HDF5)")
#     parser.add_argument("--test_data", type=str, default="data/test_data.h5", help="测试数据输出路径(HDF5)")
#
#     # 预处理参数
#     parser.add_argument("--max_instr_length", type=int, default=20, help="指令最大长度")
#     parser.add_argument("--max_instr_count", type=int, default=20, help="样本最大指令数量")
#     parser.add_argument("--vocab_size", type=int, default=2000, help="词汇表大小")
#
#     # 输出参数
#     parser.add_argument("--output_dir", type=str, default="data", help="输出目录")
#     parser.add_argument("--experiment_name", type=str, default="default", help="实验名称")
#
#     # 其他参数
#     parser.add_argument("--seed", type=int, default=42, help="随机种子")
#
#     args = parser.parse_args()
#
#     # 设置随机种子
#     set_seed(args.seed)
#
#     # 创建输出目录
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     # 创建配置
#     config = get_config(
#         model_type="transformer",  # 默认使用transformer模型
#         max_instr_length=args.max_instr_length,
#         max_instr_count=args.max_instr_count,
#         vocab_size=args.vocab_size,
#         raw_data_path=args.raw_data,
#         processed_data_path=args.processed_data,
#         train_data_path=args.train_data,
#         val_data_path=args.val_data,
#         test_data_path=args.test_data,
#         output_dir=args.output_dir,
#         experiment_name=args.experiment_name,
#         seed=args.seed
#     )
#
#     # 保存配置
#     config_path = os.path.join(args.output_dir, "preprocess_config.json")
#     config.save(config_path)
#     print(f"配置已保存到: {config_path}")
#
#     # 创建数据处理器
#     processor = RISCVDataProcessor(config)
#
#     # 处理原始数据
#     processed_data = processor.process_raw_data()
#     print(f"已处理 {len(processed_data)} 条样本")
#
#     # 划分训练集、验证集和测试集（这里用户可能已经有自己的划分方式）
#     # 这里提供一个简单的划分方法，用户可以根据需要修改
#     train_ratio = 0.7
#     val_ratio = 0.15
#     test_ratio = 0.15
#
#     import random
#     random.seed(args.seed)
#     random.shuffle(processed_data)
#
#     total_count = len(processed_data)
#     train_count = int(total_count * train_ratio)
#     val_count = int(total_count * val_ratio)
#
#     train_data = processed_data[:train_count]
#     val_data = processed_data[train_count:train_count + val_count]
#     test_data = processed_data[train_count + val_count:]
#
#     print(f"数据划分: 训练集 {len(train_data)}条, 验证集 {len(val_data)}条, 测试集 {len(test_data)}条")
#
#     # 保存划分的JSON数据
#     with open(os.path.join(args.output_dir, "train_data.json"), 'w') as f:
#         json.dump(train_data, f, indent=2)
#
#     with open(os.path.join(args.output_dir, "val_data.json"), 'w') as f:
#         json.dump(val_data, f, indent=2)
#
#     with open(os.path.join(args.output_dir, "test_data.json"), 'w') as f:
#         json.dump(test_data, f, indent=2)
#
#     # 转换为HDF5格式
#     processor.process_new_data(train_data, args.train_data, update_vocab=True)
#     processor.process_new_data(val_data, args.val_data, update_vocab=False)
#     processor.process_new_data(test_data, args.test_data, update_vocab=False)
#
#     print("数据预处理完成!")
#     print(f"训练数据: {args.train_data}")
#     print(f"验证数据: {args.val_data}")
#     print(f"测试数据: {args.test_data}")
#
#
# if __name__ == "__main__":
#     main()

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
    parser.add_argument("--max_instr_count", type=int, default=20, help="样本最大指令数量")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data", help="输出目录")
    parser.add_argument("--experiment_name", type=str, default="default", help="实验名称")

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
        processed_data_path=args.processed_data,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
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
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    import random
    random.seed(args.seed)
    random.shuffle(processed_data)

    total_count = len(processed_data)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    train_data = processed_data[:train_count]
    val_data = processed_data[train_count:train_count + val_count]
    test_data = processed_data[train_count + val_count:]

    print(f"数据划分: 训练集 {len(train_data)}条, 验证集 {len(val_data)}条, 测试集 {len(test_data)}条")

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

    # 创建HDF5文件的函数
    def create_hdf5(data, output_path):
        num_samples = len(data)
        X = np.zeros((num_samples, args.max_instr_count, args.max_instr_length), dtype=np.int32)
        instruction_counts = np.zeros((num_samples,), dtype=np.int32)
        Y = np.zeros((num_samples,), dtype=np.float32)

        for i, item in enumerate(data):
            # 填充编码的指令
            for j, encoded in enumerate(item["encoded"][:args.max_instr_count]):
                X[i, j, :len(encoded)] = encoded[:args.max_instr_length]

            instruction_counts[i] = min(item["num_instructions"], args.max_instr_count)
            Y[i] = item["throughput"]

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
            f.attrs['vocab_size'] = len(tokenizer.vocab)

        print(f"已创建HDF5文件: {output_path}")

    # 创建HDF5文件
    create_hdf5(train_data, args.train_data)
    create_hdf5(val_data, args.val_data)
    create_hdf5(test_data, args.test_data)

    print("数据预处理完成!")


if __name__ == "__main__":
    main()