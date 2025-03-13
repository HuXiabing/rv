#!/usr/bin/env python
"""
Combined data processing script for RISC-V instruction throughput data.
This script combines the functionality of:
- json_gen.py: Initial data processing from ASM and cycle data
- preprocess.py: Tokenization, encoding, and data splitting
- incremental_preprocess.py: Handling incremental data additions

The script supports two modes:
1. Full processing: Process data from scratch and create train/val splits
2. Incremental: Add new data to existing processed data, keeping validation set unchanged
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

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.tokenizer import RISCVTokenizer
from utils import set_seed


# ===== Functions from json_gen.py =====
def parse_throughput(line):
    """Extract throughput value from a line"""
    if line.startswith("Cycle Min:"):
        return float(line.split("Cycle Min:")[1].strip())
    return None


def read_instructions(file_path):
    """Read instruction content from a block file"""
    with open(file_path, "r") as f:
        lines = f.readlines()
    instructions = [line.strip() for line in lines if line.strip()]
    return instructions


def process_directory(cycle_dir, asm_dir):
    """Process all files in the specified directory"""
    results = []

    for file_name in os.listdir(cycle_dir):
        file_path = os.path.join(cycle_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        # Read file and extract throughput
        with open(file_path, "r") as f:
            lines = f.readlines()
        if len(lines) < 2:
            continue

        throughput = parse_throughput(lines[-2].strip())
        if throughput is None:
            # print(f"Error: None throughput in file {file_name}")
            continue

        if throughput == 0:
            print(f"Error: Zero throughput in file {file_name}")
            continue
        if throughput > 100:
            print(f"Warning: throughput > 100 in file {file_name}")

        # Parse filename to find corresponding block file
        if not file_name.endswith(".txt"):
            continue
        block_file_name = file_name[:-len(".txt")]
        block_file_path = Path(asm_dir) / block_file_name

        if not block_file_path.exists():
            print(f"Warning: Block file {block_file_path} not found.")
            continue

        # Read instructions from block file
        instructions = read_instructions(block_file_path)

        # Add to results
        results.append({
            "instructions": instructions,
            "throughput": throughput
        })

    return results


def generate_json(asm_dirs, cycle_dirs):
    """Generate JSON data from ASM and cycle directories"""
    all_results = []

    for asm_dir, cycle_dir in zip(asm_dirs, cycle_dirs):
        print(f"Processing ASM dir: {asm_dir}, Cycle dir: {cycle_dir}")
        results = process_directory(cycle_dir, asm_dir)
        all_results.extend(results)
        print(f"  Found {len(results)} samples")

    return all_results


# ===== Functions from preprocess.py and incremental_preprocess.py =====
def process_data(raw_data, tokenizer, max_instr_length, max_instr_count):
    """Process raw data to add tokenization and encoding"""
    processed_data = []

    for item in tqdm(raw_data, desc="Processing data"):
        instructions = item["instructions"]
        throughput = item["throughput"]

        tokenized_instructions = []
        for instr in instructions:
            tokenized = tokenizer.tokenize_instruction(instr)
            tokenized_instructions.append(tokenized)

        valid = 1
        encoded_instructions = []
        for tokenized in tokenized_instructions:
            encoded = [tokenizer.vocab.get(token, tokenizer.vocab.get('<PAD>', 0)) for token in tokenized]
            if encoded[0] in list(range(73,229 + 1)):
                encoded_instructions.append(encoded)
            else:
                valid = 0
                print(f"Warning: Invalid instruction: {tokenized}")

        if valid == 0:
            continue

        # Create a processed sample
        processed_item = {
            "instructions": instructions,
            "tokenized": tokenized_instructions,
            "encoded": encoded_instructions,
            "throughput": throughput,
            "num_instructions": len(instructions)
        }

        processed_data.append(processed_item)

    return processed_data


def get_encoded_key(item):
    """Convert encoded instructions to a string for hashing"""
    # Convert to JSON string for consistent representation
    return json.dumps(item["encoded"])


def deduplicate_data(data):
    """Deduplicate data based on the 'encoded' field"""
    seen = set()
    deduplicated = []

    for item in data:
        key = get_encoded_key(item)

        if key not in seen:
            seen.add(key)
            deduplicated.append(item)

    print(f"Deduplicated from {len(data)} to {len(deduplicated)} samples")
    return deduplicated


def split_data(data, val_ratio, train_ratio, seed=42):
    """Split data into training and validation sets"""
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)

    total_count = len(data_copy)
    val_count = int(total_count * val_ratio)

    val_data = data_copy[:val_count]
    train_data = data_copy[val_count:]

    print(f"Split data: Validation set: {len(val_data)}, Training set: {len(train_data)}")
    return train_data, val_data


def create_hdf5(data, output_path, max_instr_count, max_instr_length):
    """Create HDF5 file from processed data"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_samples = len(data)
    X = np.zeros((num_samples, max_instr_count, max_instr_length), dtype=np.int32)  # [batch_size, sequence_length, embedding_dim]
    instruction_counts = np.zeros(num_samples, dtype=np.int32)
    Y = np.zeros((num_samples,), dtype=np.float32)

    for i, item in enumerate(data):
        for j, encoded in enumerate(item["encoded"][:max_instr_count]):
            X[i, j, :len(encoded)] = encoded[:max_instr_length]

        instruction_counts[i] = min(item["num_instructions"], max_instr_count)
        Y[i] = item["throughput"]

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('X', data=X, compression='gzip')
        f.create_dataset('instruction_counts', data=instruction_counts)
        f.create_dataset('Y', data=Y)

        dt = h5py.special_dtype(vlen=str)
        instr_text = np.array([json.dumps(item["instructions"]) for item in data], dtype=dt)
        f.create_dataset('instruction_text', data=instr_text)

        f.attrs['num_samples'] = num_samples
        f.attrs['max_instr_count'] = max_instr_count
        f.attrs['max_instr_length'] = max_instr_length

    print(f"HDF5 file created: {output_path}")


# ===== Function for incremental processing =====
def incremental_update(existing_train_data, existing_val_data, new_processed_data):
    """
    Update training data with new data while keeping validation set unchanged

    This function ensures:
    1. The validation set remains unchanged
    2. New data is added to the training set
    3. Duplicates are removed from the training set
    4. Any samples in validation are not duplicated in training
    """
    # Create a set of encoded keys from validation data
    val_encoded_keys = {get_encoded_key(item) for item in existing_val_data}

    # Create a set of existing training encoded keys
    train_encoded_keys = {get_encoded_key(item) for item in existing_train_data}

    # Filter new data to remove any duplicates with validation set
    filtered_new_data = []
    for item in new_processed_data:
        key = get_encoded_key(item)
        if key not in val_encoded_keys and key not in train_encoded_keys:
            filtered_new_data.append(item)
            train_encoded_keys.add(key)  # Update to avoid duplicates within new data

    # Combine existing and filtered new data
    updated_train_data = existing_train_data + filtered_new_data

    print(f"Added {len(filtered_new_data)} unique new samples to training data")
    print(f"Updated training data size: {len(updated_train_data)}")

    return updated_train_data, existing_val_data


def main():
    parser = argparse.ArgumentParser(description="RISC-V Instruction Throughput Data Processing")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["full", "incremental"], default="full",
                        help="Processing mode: full (process from scratch) or incremental (add new data)")

    # Inputs for JSON generation
    parser.add_argument("--asm_dirs", nargs="+", required=True,
                        help="List of directories containing ASM files")
    parser.add_argument("--cycle_dirs", nargs="+", required=True,
                        help="List of directories containing cycle measurement files")

    # Existing data for incremental mode
    parser.add_argument("--existing_processed_json", type=str, default="data/processed_data.json",
                        help="Path to the existing processed JSON file (for incremental mode)")
    parser.add_argument("--existing_train_json", type=str, default="data/train_data.json",
                        help="Path to the existing train JSON file (for incremental mode)")
    parser.add_argument("--existing_val_json", type=str, default="data/val_data.json",
                        help="Path to the existing validation JSON file (for incremental mode)")
    parser.add_argument("--existing_train_h5", type=str, default="data/train_data.h5",
                        help="Path to the existing train HDF5 file (for incremental mode)")
    parser.add_argument("--existing_val_h5", type=str, default="data/val_data.h5",
                        help="Path to the existing validation HDF5 file (for incremental mode)")

    # Output paths
    # parser.add_argument("--raw_json", type=str, default="data/raw_data.json",
    #                     help="Output path for raw JSON data")
    parser.add_argument("--processed_json", type=str, default="data/processed_data.json",
                        help="Output path for processed JSON data")
    parser.add_argument("--train_json", type=str, default="data/train_data.json",
                        help="Output path for training JSON data")
    parser.add_argument("--val_json", type=str, default="data/val_data.json",
                        help="Output path for validation JSON data")
    parser.add_argument("--train_h5", type=str, default="data/train_data.h5",
                        help="Output path for training HDF5 data")
    parser.add_argument("--val_h5", type=str, default="data/val_data.h5",
                        help="Output path for validation HDF5 data")

    # Processing parameters
    parser.add_argument("--max_instr_length", type=int, default=8, help="Maximum length of an instruction")
    parser.add_argument("--max_instr_count", type=int, default=400, help="Maximum number of instructions per sample")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of the validation set")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")

    # Others
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Check inputs
    if len(args.asm_dirs) != len(args.cycle_dirs):
        raise ValueError("Number of ASM directories must match number of cycle directories")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tokenizer
    tokenizer = RISCVTokenizer(max_instr_length=args.max_instr_length)

    # Generate raw JSON data
    print("Generating raw JSON data...")
    raw_data = generate_json(args.asm_dirs, args.cycle_dirs)

    # Save raw data
    # with open(args.raw_json, 'w') as f:
    #     json.dump(raw_data, f, indent=2)
    # print(f"Raw data saved to {args.raw_json} ({len(raw_data)} samples)")

    # Process data - tokenize and encode
    print("Processing data...")
    processed_data = process_data(raw_data, tokenizer, args.max_instr_length, args.max_instr_count)

    if args.mode == "full":
        # Full processing mode
        # Deduplicate processed data
        processed_data = deduplicate_data(processed_data)

        # Save processed data to JSON
        with open(args.processed_json, 'w') as f:
            json.dump(processed_data, f, indent=2)
        print(f"Processed data saved to {args.processed_json}")

        # Split data into train and validation sets
        train_data, val_data = split_data(processed_data, args.val_ratio, 1.0 - args.val_ratio, args.seed)

        # Save split data to JSON
        with open(args.train_json, 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(args.val_json, 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"Train data saved to {args.train_json}")
        print(f"Validation data saved to {args.val_json}")

        # Create HDF5 files
        create_hdf5(train_data, args.train_h5, args.max_instr_count, args.max_instr_length)
        create_hdf5(val_data, args.val_h5, args.max_instr_count, args.max_instr_length)

    else:
        # Incremental mode
        # Load existing data
        print("Loading existing processed data...")
        with open(args.existing_processed_json, 'r') as f:
            existing_processed_data = json.load(f)

        with open(args.existing_train_json, 'r') as f:
            existing_train_data = json.load(f)

        with open(args.existing_val_json, 'r') as f:
            existing_val_data = json.load(f)

        # Merge processed data with existing processed data
        all_processed_data = existing_processed_data + processed_data
        deduplicated_processed_data = deduplicate_data(all_processed_data)

        # Save merged processed data
        with open(args.processed_json, 'w') as f:
            json.dump(deduplicated_processed_data, f, indent=2)
        print(f"Updated processed data saved to {args.processed_json}")

        # Update training data while keeping validation set unchanged
        updated_train_data, val_data = incremental_update(
            existing_train_data, existing_val_data, processed_data
        )

        # Save updated training data
        with open(args.train_json, 'w') as f:
            json.dump(updated_train_data, f, indent=2)
        print(f"Updated train data saved to {args.train_json}")

        # Create HDF5 files
        create_hdf5(updated_train_data, args.train_h5, args.max_instr_count, args.max_instr_length)
        create_hdf5(val_data, args.val_h5, args.max_instr_count, args.max_instr_length)

    print("Data processing completed!")


if __name__ == "__main__":
    main()