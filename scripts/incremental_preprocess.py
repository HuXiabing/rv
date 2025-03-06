#!/usr/bin/env python
"""
Incremental data preprocessing script -
Convert new raw JSON data into processed JSON and HDF5 formats, and merge it into existing data.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import get_config
from data.tokenizer import RISCVTokenizer
from utils import set_seed

def merge_json_files(existing_json_path, new_json_path, output_json_path):

    with open(existing_json_path, 'r') as f:
        existing_data = json.load(f)

    with open(new_json_path, 'r') as f:
        new_data = json.load(f)

    merged_data = existing_data + new_data

    with open(output_json_path, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Merged JSON data saved to: {output_json_path}")

def merge_h5_files(existing_h5_path, new_h5_path, output_h5_path):

    with h5py.File(existing_h5_path, 'r') as f1:
        X1 = f1['X'][:]
        instruction_counts1 = f1['instruction_counts'][:]
        Y1 = f1['Y'][:]
        has_instruction_text = 'instruction_text' in f1
        if has_instruction_text:
            instruction_text1 = f1['instruction_text'][:]

        num_samples1 = f1.attrs['num_samples']
        max_instr_count = f1.attrs.get('max_instr_count', 20)
        max_instr_length = f1.attrs.get('max_instr_length', 8)

    with h5py.File(new_h5_path, 'r') as f2:
        X2 = f2['X'][:]
        instruction_counts2 = f2['instruction_counts'][:]
        Y2 = f2['Y'][:]
        if has_instruction_text:
            instruction_text2 = f2['instruction_text'][:]

        num_samples2 = f2.attrs['num_samples']

    X_merged = np.vstack([X1, X2])
    instruction_counts_merged = np.concatenate([instruction_counts1, instruction_counts2])
    Y_merged = np.concatenate([Y1, Y2])

    if has_instruction_text:
        instruction_text_merged = np.concatenate([instruction_text1, instruction_text2])

    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    with h5py.File(output_h5_path, 'w') as f_out:

        f_out.create_dataset('X', data=X_merged, compression='gzip')
        f_out.create_dataset('instruction_counts', data=instruction_counts_merged)
        f_out.create_dataset('Y', data=Y_merged)

        if has_instruction_text:

            dt = h5py.special_dtype(vlen=str)
            f_out.create_dataset('instruction_text', data=instruction_text_merged, dtype=dt)

        f_out.attrs['num_samples'] = num_samples1 + num_samples2
        f_out.attrs['max_instr_count'] = max_instr_count
        f_out.attrs['max_instr_length'] = max_instr_length

    print(f"Merged HDF5 file saved to: {output_h5_path}")

def main():
    parser = argparse.ArgumentParser(description="RISC-V Instruction Throughput Incremental Data Preprocessing")

    parser.add_argument("--raw_data", type=str, required=True, help="Path to the new raw JSON data file")
    parser.add_argument("--existing_train_json", type=str, default="data/train_data.json",
                        help="Path to the existing training JSON data file")
    parser.add_argument("--existing_train_h5", type=str, default="data/train_data.h5",
                        help="Path to the existing training HDF5 data file")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")

    parser.add_argument("--max_instr_length", type=int, default=8, help="Maximum length of an instruction")
    parser.add_argument("--max_instr_count", type=int, default=400, help="Maximum number of instructions per sample")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    set_seed(args.seed)
    # Create output directory
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

        processed_item = {
            "instructions": instructions,
            "tokenized": tokenized_instructions,
            "encoded": encoded_instructions,
            "throughput": throughput,
            "num_instructions": len(instructions)
        }

        processed_data.append(processed_item)

    new_json_path = os.path.join(args.output_dir, "new_processed_data.json")
    print(f"Save processed data to JSON file: {new_json_path}")
    with open(new_json_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    merged_json_path = os.path.join(args.output_dir, "incremental_train.json")
    merge_json_files(args.existing_train_json, new_json_path, merged_json_path)

    new_h5_path = os.path.join(args.output_dir, "new_train_data.h5")
    # print("Creating h5 file....")

    def create_hdf5(data, output_path):
        num_samples = len(data)
        X = np.zeros((num_samples, args.max_instr_count, args.max_instr_length), dtype=np.int32)
        instruction_counts = np.zeros(num_samples, dtype=np.int32)
        Y = np.zeros((num_samples,), dtype=np.float32)

        for i, item in enumerate(data):
            for j, encoded in enumerate(item["encoded"][:args.max_instr_count]):
                X[i, j, :len(encoded)] = encoded[:args.max_instr_length]

            instruction_counts[i] = min(item["num_instructions"], args.max_instr_count)
            Y[i] = item["throughput"]

        # print(f"Creating h5 file: {output_path}")
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

    create_hdf5(processed_data, new_h5_path)

    merged_h5_path = os.path.join(args.output_dir, "incremental_train.h5")
    merge_h5_files(args.existing_train_h5, new_h5_path, merged_h5_path)

    print("Incremental data preprocessing completed!")


if __name__ == "__main__":
    main()