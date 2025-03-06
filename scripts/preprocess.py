# !/usr/bin/env python
"""
data preprocess script - convert raw JSON data to processed JSON format and HDF5 format
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import random
# add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import get_config
from data.tokenizer import RISCVTokenizer
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="RISC-V Instruction Throughput Data Preprocessing")

    # data arguments
    parser.add_argument("--raw_data", type=str, required=True, help="Path to the raw JSON data file")
    parser.add_argument("--processed_data", type=str, default="data/processed_data.json",
                        help="Path to the processed JSON data file")
    parser.add_argument("--train_data", type=str, default="data/train_data.h5",
                        help="Output path for training data (HDF5)")
    parser.add_argument("--val_data", type=str, default="data/val_data.h5",
                        help="Output path for validation data (HDF5)")
    parser.add_argument("--test_data", type=str, default="data/test_data.h5",
                        help="Output path for testing data (HDF5)")

    # preprocess arguments
    parser.add_argument("--max_instr_length", type=int, default=8, help="Maximum length of an instruction")
    parser.add_argument("--max_instr_count", type=int, default=400, help="Maximum number of instructions per sample")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of the training set")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of the validation set")
    parser.add_argument("--test_ratio", type=float, default=0, help="Proportion of the testing set")

    # output arguments
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")

    # others
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()

    # check whether the total ratio is illegal
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(f"total ratio must be 1.0, current is {total_ratio}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = RISCVTokenizer(max_instr_length=args.max_instr_length)

    print(f"Loading raw data: {args.raw_data}")
    with open(args.raw_data, 'r') as f:
        raw_data = json.load(f)

    processed_data = []
    for item in tqdm(raw_data, desc="Processing data"):
        instructions = item["instructions"]
        throughput = item["throughput"]

        tokenized_instructions = []
        for instr in instructions:
            tokenized = tokenizer.tokenize_instruction(instr)
            tokenized_instructions.append(tokenized)

        encoded_instructions = []
        for tokenized in tokenized_instructions:
            encoded = [tokenizer.vocab.get(token, tokenizer.vocab.get('<PAD>', 0)) for token in tokenized]
            encoded_instructions.append(encoded)

        # create a processed sample
        processed_item = {
            "instructions": instructions,  # raw instructions
            "tokenized": tokenized_instructions,  # tokenized instructions
            "encoded": encoded_instructions,  # encoded instructions
            "throughput": throughput,  # throughput
            "num_instructions": len(instructions)  # instruction counts of every single sample
        }

        processed_data.append(processed_item)

    print(f"Save processed data to JSON file: {args.processed_data}")
    with open(args.processed_data, 'w') as f:
        json.dump(processed_data, f, indent=2)

    # print("Creating h5 file....")
    random.seed(args.seed)
    random.shuffle(processed_data)

    total_count = len(processed_data)
    val_count = int(total_count * args.val_ratio)
    train_count = int(total_count * args.train_ratio)
    # test_count = total_count - val_count - train_count

    val_data = processed_data[:val_count]
    train_data = processed_data[val_count:val_count + train_count]
    test_data = processed_data[val_count + train_count:]

    print(f"Validation set {len(val_data)}, Training set {len(train_data)}, Test set {len(test_data)}")

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
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        num_samples = len(data)
        X = np.zeros((num_samples, args.max_instr_count, args.max_instr_length), dtype=np.int32) #encoded matrix
        instruction_counts = np.zeros(num_samples, dtype=np.int32)
        Y = np.zeros((num_samples,), dtype=np.float32)

        for i, item in enumerate(data):
            for j, encoded in enumerate(item["encoded"][:args.max_instr_count]):
                X[i, j, :len(encoded)] = encoded[:args.max_instr_length]

            instruction_counts[i] = min(item["num_instructions"], args.max_instr_count)
            Y[i] = item["throughput"]

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('instruction_counts', data=instruction_counts)
            f.create_dataset('Y', data=Y)

            dt = h5py.special_dtype(vlen=str)
            instr_text = np.array([json.dumps(item["instructions"]) for item in data], dtype=dt)
            f.create_dataset('instruction_text', data=instr_text)

            f.attrs['num_samples'] = num_samples
            f.attrs['max_instr_count'] = args.max_instr_count
            f.attrs['max_instr_length'] = args.max_instr_length

        print(f"HDF5 file created: {output_path}")

    create_hdf5(val_data, args.val_data)
    create_hdf5(train_data, args.train_data)
    create_hdf5(test_data, args.test_data)

    print("Data preprocessing completed!")


if __name__ == "__main__":
    main()