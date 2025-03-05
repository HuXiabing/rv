import os
import json
import h5py
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from .tokenizer import RISCVTokenizer

class RISCVDataProcessor:
    """处理RISC-V指令数据集"""

    def __init__(self, config):
        """
        初始化数据处理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.tokenizer = RISCVTokenizer(
            vocab_size=config.vocab_size,
            max_instr_length=config.max_instr_length
        )

    def process_raw_data(self):
        """处理原始数据并保存为中间JSON格式"""
        print("处理原始RISC-V指令数据...")

        # 加载原始数据
        with open(self.config.raw_data_path, 'r') as f:
            raw_data = json.load(f)

        # 提取所有指令列表，用于构建词汇表
        all_instructions_lists = [item["instructions"] for item in raw_data]

        # 构建词汇表
        self.tokenizer.build_vocab_from_instructions(all_instructions_lists)

        # 处理数据
        processed_data = []
        for item in tqdm(raw_data, desc="处理数据"):
            instructions = item["instructions"]
            throughput = item["throughput"]

            # 对指令进行分词
            tokenized_instructions = []
            for instr in instructions:
                # 使用自定义分词器处理单条指令
                tokenized = self.tokenizer.tokenize_instruction(instr)
                tokenized_instructions.append(tokenized)

            # 将分词结果编码为token ID
            encoded_instructions = []
            for tokenized in tokenized_instructions:
                encoded = [self.tokenizer.vocab.get(token, self.tokenizer.vocab.get('<PAD>', 0)) for token in tokenized]
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
        os.makedirs(os.path.dirname(self.config.processed_data_path), exist_ok=True)
        with open(self.config.processed_data_path, 'w') as f:
            json.dump(processed_data, f, indent=2)

        print(f"已处理 {len(processed_data)} 条样本，保存到 {self.config.processed_data_path}")
        return processed_data

    def process_data_to_h5(self,
                           input_path: Optional[str] = None,
                           output_path: Optional[str] = None,
                           is_training: bool = True) -> str:
        """
        将处理后的JSON数据转换为HDF5格式

        Args:
            input_path: 输入JSON文件路径，默认使用config中的路径
            output_path: 输出HDF5文件路径，默认使用config中的路径
            is_training: 是否为训练集，如果是，则更新词汇表

        Returns:
            输出HDF5文件路径
        """
        input_path = input_path or self.config.processed_data_path
        if output_path is None:
            output_path = self.config.train_data_path if is_training else self.config.val_data_path

        print(f"转换数据为HDF5格式 ({output_path})...")

        # 确保词汇表已加载
        if not hasattr(self.tokenizer, 'vocab') or not self.tokenizer.vocab:
            if os.path.exists(self.config.vocab_path):
                self.tokenizer.load_vocab(self.config.vocab_path)
            else:
                raise ValueError("词汇表不存在，请先调用process_raw_data或手动加载词汇表")

        # 加载处理后的JSON数据
        with open(input_path, 'r') as f:
            processed_data = json.load(f)

        # 准备数据
        num_samples = len(processed_data)
        X = np.zeros((num_samples, self.config.max_instr_count, self.config.max_instr_length), dtype=np.int32)
        instruction_counts = np.zeros((num_samples,), dtype=np.int32)
        Y = np.zeros((num_samples,), dtype=np.float32)

        for i, item in enumerate(tqdm(processed_data, desc="编码样本")):
            # 填充编码的指令
            for j, encoded in enumerate(item["encoded"][:self.config.max_instr_count]):
                X[i, j, :len(encoded)] = encoded[:self.config.max_instr_length]

            instruction_counts[i] = min(item["num_instructions"], self.config.max_instr_count)
            Y[i] = item["throughput"]

        # 创建HDF5文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('instruction_counts', data=instruction_counts)
            f.create_dataset('Y', data=Y)

            # 存储原始指令文本
            dt = h5py.special_dtype(vlen=str)
            instr_text = np.array([json.dumps(item["instructions"]) for item in processed_data], dtype=dt)
            f.create_dataset('instruction_text', data=instr_text)

            # 存储元数据
            f.attrs['num_samples'] = num_samples
            f.attrs['max_instr_count'] = self.config.max_instr_count
            f.attrs['max_instr_length'] = self.config.max_instr_length
            f.attrs['vocab_size'] = self.tokenizer.current_vocab_size

        print(f"已将数据转换为HDF5格式，保存到 {output_path}")
        return output_path

    def _encode_sample(self, instructions: List[str]) -> List[List[int]]:
        """
        编码单个样本的所有指令

        Args:
            instructions: 指令列表

        Returns:
            编码后的token ID二维列表
        """
        encoded_instructions = []

        # 对每条指令进行编码
        for instruction in instructions[:self.config.max_instr_count]:
            encoded = self.tokenizer.encode_instruction(instruction)
            encoded_instructions.append(encoded)

        # 补充空指令，确保每个样本具有相同数量的指令
        pad_instruction = [self.tokenizer.vocab["<PAD>"]] * self.config.max_instr_length
        while len(encoded_instructions) < self.config.max_instr_count:
            encoded_instructions.append(pad_instruction)

        return encoded_instructions

    def process_new_data(self,
                         input_data: List[Dict[str, Any]],
                         output_path: str,
                         update_vocab: bool = False) -> str:
        """
        处理新数据（用于推理或二次训练）

        Args:
            input_data: 输入数据列表
            output_path: 输出HDF5文件路径
            update_vocab: 是否更新词汇表

        Returns:
            输出HDF5文件路径
        """
        # 确保词汇表已加载
        if not hasattr(self.tokenizer, 'vocab') or not self.tokenizer.vocab:
            if os.path.exists(self.config.vocab_path):
                self.tokenizer.load_vocab(self.config.vocab_path)
            else:
                raise ValueError("词汇表不存在，请先调用process_raw_data或手动加载词汇表")

        # 如果需要更新词汇表
        if update_vocab:
            all_instructions_lists = [item["instructions"] for item in input_data]
            self.tokenizer.build_vocab_from_instructions(all_instructions_lists)
            self.tokenizer.save_vocab(self.config.vocab_path)

        # 准备数据
        num_samples = len(input_data)
        X = np.zeros((num_samples, self.config.max_instr_count, self.config.max_instr_length), dtype=np.int32)
        instruction_counts = np.zeros((num_samples,), dtype=np.int32)
        Y = np.zeros((num_samples,), dtype=np.float32)

        for i, item in enumerate(tqdm(input_data, desc="编码样本")):
            # 编码指令
            encoded_instructions = self._encode_sample(item["instructions"])
            X[i] = encoded_instructions

            # 记录指令数量
            instruction_counts[i] = len(item["instructions"])

            # 记录吞吐量（如果存在）
            if "throughput" in item:
                Y[i] = item["throughput"]

        # 创建HDF5文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('instruction_counts', data=instruction_counts)
            f.create_dataset('Y', data=Y)

            # 存储元数据
            f.attrs['num_samples'] = num_samples
            f.attrs['max_instr_count'] = self.config.max_instr_count
            f.attrs['max_instr_length'] = self.config.max_instr_length
            f.attrs['vocab_size'] = self.tokenizer.current_vocab_size

        return output_path

