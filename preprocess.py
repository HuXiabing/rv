#!/usr/bin/env python
"""
Combined data processing script for RISC-V instruction throughput data.

The script supports two modes:
1. Full processing: Process data from scratch and create train/val splits
2. Incremental: Add new data to existing processed data, keeping validation set unchanged
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import random

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.tokenizer import RISCVTokenizer
from utils import set_seed


# ===== Functions from json_gen.py =====
def parse_throughput(line):
    """Extract throughput value from a line"""
    if line.startswith("Cycle Mode:"):
        throughput = float(line.split("Cycle Mode:")[1].strip())
        return throughput
    return None

def parse_throughput_mca(line):
    """Extract throughput value from mca"""
    if line.startswith("Cycles Per Block:"):
        throughput = float(line.split("Cycles Per Block:")[1].strip())
        return throughput
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

        # throughput = parse_throughput(lines[-2].strip())
        throughput = parse_throughput_mca(lines[10].strip())

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
        block_file_name = file_name[:-len(".txt")] # + ".S"
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


# def generate_json(asm_dirs, cycle_dirs):
#     """Generate JSON data from ASM and cycle directories"""
#     all_results = []
#
#     for asm_dir, cycle_dir in zip(asm_dirs, cycle_dirs):
#         print(f"Processing ASM dir: {asm_dir}, Cycle dir: {cycle_dir}")
#         results = process_directory(cycle_dir, asm_dir)
#         all_results.extend(results)
#         print(f"  Found {len(results)} samples")
#
#     return all_results
# def generate_json(cycle_jsons):
#     """Generate JSON data from ASM and cycle directories"""
#     all_results = []
#     for cycle_json in cycle_jsons:
#         with open(cycle_json, 'r') as f:
#             data = json.load(f)
#         for k,v in data.items():
#             for row in v['mca_result'].split("\n"):
#                 if "Block RThroughput" in row:
#                     throughput = float(row.split("Block RThroughput:")[1].strip())
#                     break
#             instructions = v['asm']
#
#             all_results.append({
#                 "instructions": instructions,
#                 "throughput": throughput})
#     return all_results
def generate_json_mca(cycle_jsons):
    """Generate JSON data from ASM and cycle directories"""
    all_results = []
    for cycle_json in cycle_jsons:
        with open(cycle_json, 'r') as f:
            data = json.load(f)
        # print(len(data))
        for entry in data:
            # print(entry['mca_result'])
            for row in entry['mca_result'].split("\n"):
                if "Cycles Per Block" in row:
                    throughput = float(row.split("Cycles Per Block:")[1].strip())
                    break
            instructions = entry['asm'].replace("\\n", "\n")
            # print(throughput)

            all_results.append({
                "instructions": instructions.split("\n"),
                "throughput": throughput})
    return all_results

def generate_json(cycle_jsons):
    """Generate JSON data from ASM and cycle directories"""
    all_results = []
    for cycle_json in cycle_jsons:
        with open("random_generate/starfive/" + cycle_json, 'r') as f:
            data = json.load(f)
        for entry in data:
            # print(entry['mca_result'])
            throughput = None
            for row in entry['result'].split("\n"):
                # print(row)
                if "Cycle Mode:" in row:
                    throughput = float(row.split("Cycle Mode:")[1].strip())
                    break
            if throughput == None:
                continue
            instructions = entry['asm'].replace("\\n", "\n")
            # print(throughput)

            all_results.append({
                "instructions": instructions.split("\n"),
                "throughput": throughput})
    return all_results
# ===== Functions from preprocess.py and incremental_preprocess.py =====
def process_data(raw_data, tokenizer, seed = 71, shuffle = True): # 4516
    """Process raw data to add tokenization and encoding"""
    processed_data = []

    for idx, item in enumerate(tqdm(raw_data, desc="Processing data")):
        instructions = item["instructions"]
        throughput = item["throughput"]

        tokenized_instructions = []
        for instr in instructions:
            instr = instr.replace("\\t", "\t")
            tokenized = tokenizer.tokenize_instruction(instr)
            tokenized_instructions.append(tokenized)

        valid = 1
        encoded_instructions = []
        for tokenized in tokenized_instructions:
            encoded = [tokenizer.vocab.get(token, tokenizer.vocab.get('<PAD>', 0)) for token in tokenized]
            if encoded[0] in list(range(73,137 + 1)):
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
            "num_instructions": len(instructions),
            "id": idx,
        }

        processed_data.append(processed_item)
    processed_data_copy = processed_data.copy()
    if shuffle == True:
        random.seed(seed)
        random.shuffle(processed_data_copy)
    return processed_data_copy


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
    # random.seed(seed)
    data_copy = data.copy()
    # random.shuffle(data_copy)

    total_count = len(data_copy)
    val_count = int(total_count * val_ratio)

    val_data = data_copy[:val_count]
    train_data = data_copy[val_count:]

    print(f"Split data: Validation set: {len(val_data)}, Training set: {len(train_data)}")
    return train_data, val_data


def split_data_by_count(data, train_samples, val_samples, seed=42):
    """
    Split data into training and validation sets based on exact sample counts.

    If total samples are insufficient, prioritize validation set requirements.
    """
    # random.seed(seed)
    data_copy = data.copy()
    # random.shuffle(data_copy)

    total_count = len(data_copy)
    requested_total = train_samples + val_samples

    # Case 1: We have enough samples to satisfy both requirements
    if total_count >= requested_total:
        val_data = data_copy[:val_samples]
        train_data = data_copy[val_samples:val_samples + train_samples]
        print(f"Generated requested sample counts: Validation set: {len(val_data)}, Training set: {len(train_data)}")

    # Case 2: Not enough samples for both, prioritize validation set
    else:
        actual_val_samples = min(val_samples, total_count)
        actual_train_samples = max(0, total_count - actual_val_samples)

        val_data = data_copy[:actual_val_samples]
        train_data = data_copy[actual_val_samples:] if actual_train_samples > 0 else []

        print(f"Warning: Insufficient samples to meet requested counts")
        print(f"Generated sample counts: Validation set: {len(val_data)}, Training set: {len(train_data)}")
        print(f"Requested sample counts: Validation set: {val_samples}, Training set: {train_samples}")

    return train_data, val_data

# ===== Function for incremental processing =====
# def incremental_update(existing_train_data, existing_val_data, new_processed_data):
#     """
#     Update training data with new data while keeping validation set unchanged
#
#     This function ensures:
#     1. The validation set remains unchanged
#     2. New data is added to the training set
#     3. Duplicates are removed from the training set
#     4. Any samples in validation are not duplicated in training
#     """
#     # Create a set of encoded keys from validation data
#     val_encoded_keys = {get_encoded_key(item) for item in existing_val_data}
#
#     # Create a set of existing training encoded keys
#     train_encoded_keys = {get_encoded_key(item) for item in existing_train_data}
#
#     # Filter new data to remove any duplicates with validation set
#     filtered_new_data = []
#     for item in new_processed_data:
#         key = get_encoded_key(item)
#         if key not in val_encoded_keys and key not in train_encoded_keys:
#             filtered_new_data.append(item)
#             train_encoded_keys.add(key)  # Update to avoid duplicates within new data
#
#     # Combine existing and filtered new data
#     updated_train_data = existing_train_data + filtered_new_data
#
#     print(f"Added {len(filtered_new_data)} unique new samples to training data")
#     print(f"Updated training data size: {len(updated_train_data)}")
#
#     return updated_train_data, existing_val_data

def incremental_update(existing_train_data, existing_val_data, new_processed_data, train_samples, val_samples):
    """
    Update training data with new data while keeping validation set unchanged or expanded to meet requirements

    This function ensures:
    1. The validation set meets the requested sample count if possible
    2. New data is added to the training set to meet the requested count
    3. Duplicates are removed
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

    # First, check if we need to expand validation set
    val_data = existing_val_data.copy()
    remaining_new_data = filtered_new_data.copy()

    if len(val_data) < val_samples:
        # We need more validation samples
        needed_val_samples = val_samples - len(val_data)

        # Take samples from new data for validation
        if needed_val_samples > 0 and remaining_new_data:
            new_val_samples = min(needed_val_samples, len(remaining_new_data))
            val_data.extend(remaining_new_data[:new_val_samples])
            remaining_new_data = remaining_new_data[new_val_samples:]
            val_encoded_keys.update(get_encoded_key(item) for item in val_data[-new_val_samples:])

            print(f"Added {new_val_samples} new samples to validation data")

    # Now handle training data
    # Start with existing training data
    train_data = existing_train_data.copy()

    # Calculate how many more training samples we need
    needed_train_samples = max(0, train_samples - len(train_data))

    # Add samples from remaining new data to training
    if needed_train_samples > 0 and remaining_new_data:
        new_train_samples = min(needed_train_samples, len(remaining_new_data))
        train_data.extend(remaining_new_data[:new_train_samples])
        print(f"Added {new_train_samples} new samples to training data")

    # If we still have excess training data, trim it
    if len(train_data) > train_samples:
        train_data = train_data[:train_samples]
        print(f"Trimmed training data to {train_samples} samples")

    print(f"Final data sizes: Validation set: {len(val_data)}, Training set: {len(train_data)}")
    print(f"Requested sizes: Validation set: {val_samples}, Training set: {train_samples}")

    return train_data, val_data

def updated_training_data(existing_processed_data, processed_data):
    """Update training data with new data while keeping validation set unchanged"""
    processed_data_encoded_keys = {get_encoded_key(item) for item in existing_processed_data}

    filtered_new_data = []
    for item in processed_data:
        key = get_encoded_key(item)
        if key not in processed_data_encoded_keys:
            filtered_new_data.append(item)
    print(f"Added {len(filtered_new_data)} new samples to training data")
    return filtered_new_data

def main():
    parser = argparse.ArgumentParser(description="RISC-V Instruction Throughput Data Processing")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["full", "incremental"], default="full",
                        help="Processing mode: full (process from scratch) or incremental (add new data)")

    parser.add_argument("--cycle_jsons", nargs="+", required=True,
                        help="List of jsons containing cycle measurement files")

    # Existing data for incremental mode
    # parser.add_argument("--existing_processed_json", type=str, default="data/processed_data.json",
    #                     help="Path to the existing processed JSON file (for incremental mode)")
    parser.add_argument("--existing_train_json", type=str, default="data/train_data.json",
                        help="Path to the existing train JSON file (for incremental mode)")
    parser.add_argument("--existing_val_json", type=str, default="data/val_data.json",
                        help="Path to the existing validation JSON file (for incremental mode)")

    # Output paths
    # parser.add_argument("--processed_json", type=str, default="data/processed_data.json",
    #                     help="Output path for processed JSON data")
    parser.add_argument("--train_json", type=str, default="data/train_data.json",
                        help="Output path for training JSON data")
    parser.add_argument("--val_json", type=str, default="data/val_data.json",
                        help="Output path for validation JSON data")

    # Sample count parameters
    parser.add_argument("--train_samples", type=int, default=None,
                        help="Number of training samples to generate (if None, uses all available samples)")
    parser.add_argument("--val_samples", type=int, default=None,
                        help="Number of validation samples to generate (if None, uses val_ratio)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of the validation set")

    # Output directory
    # parser.add_argument("--output_dir", type=str, default="data", help="Output directory")

    # Others
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Check inputs
    # if len(args.asm_dirs) != len(args.cycle_dirs):
    #     raise ValueError("Number of ASM directories must match number of cycle directories")

    # set_seed(args.seed)
    # os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tokenizer
    tokenizer = RISCVTokenizer()

    # Generate raw JSON data
    print("Generating raw JSON data...")
    raw_data = generate_json_mca(args.cycle_jsons)

    # Process data - tokenize and encode
    print("Processing data...")
    processed_data = process_data(raw_data, tokenizer)

    if args.mode == "full":
        # Full processing mode
        # Deduplicate processed data
        processed_data = deduplicate_data(processed_data)
        saved_samples = args.train_samples + args.val_samples
        processed_data = processed_data[:saved_samples]
        print("saved processed data: ", len(processed_data))

        # Save processed data to JSON
        # with open(args.processed_json, 'w') as f:
        #     json.dump(processed_data, f, indent=2)
        # print(f"Processed data saved to {args.processed_json}")

        # Split data into train and validation sets
        total_samples = len(processed_data)

        if args.train_samples is None and args.val_samples is None:
            # Use val_ratio to determine split
            val_samples = int(total_samples * args.val_ratio)
            train_samples = total_samples - val_samples
        elif args.val_samples is None:
            # Train samples specified, use val_ratio for validation
            val_samples = int(min(total_samples - args.train_samples,
                                  total_samples * args.val_ratio))
            train_samples = min(args.train_samples, total_samples - val_samples)
        elif args.train_samples is None:
            # Val samples specified, use rest for training
            val_samples = min(args.val_samples, total_samples)
            train_samples = total_samples - val_samples
        else:
            # Both specified
            val_samples = args.val_samples
            train_samples = args.train_samples

        # Split data into train and validation sets
        train_data, val_data = split_data_by_count(processed_data, train_samples, val_samples, args.seed)

        # Save split data to JSON
        with open(args.train_json, 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(args.val_json, 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"Train data saved to {args.train_json}")
        print(f"Validation data saved to {args.val_json}")

    else:
        # Incremental mode
        # Load existing data
        print("Loading existing processed data...")
        # with open(args.existing_processed_json, 'r') as f:
        #     existing_processed_data = json.load(f)

        with open(args.existing_train_json, 'r') as f:
            existing_train_data = json.load(f)

        with open(args.existing_val_json, 'r') as f:
            existing_val_data = json.load(f)

        existing_processed_data = existing_train_data + existing_val_data

        # Merge processed data with existing processed data
        all_processed_data = existing_processed_data + processed_data

        deduplicated_processed_data = deduplicate_data(all_processed_data)

        # Save merged processed data
        # with open(args.processed_json, 'w') as f:
        #     json.dump(deduplicated_processed_data, f, indent=2)
        # print(f"Updated processed data saved to {args.processed_json}")

        # Determine sample counts
        if args.train_samples is None and args.val_samples is None:
            # Keep existing validation set and add new data to training
            val_samples = len(existing_val_data)
            # Don't limit training samples
            train_samples = float('inf')
        else:
            # Use specified counts
            val_samples = args.val_samples if args.val_samples is not None else len(existing_val_data)
            train_samples = args.train_samples if args.train_samples is not None else float('inf')

        # Update training and validation data
        # updated_train_data, val_data = incremental_update(
        #     existing_train_data, existing_val_data, processed_data, train_samples, val_samples
        # )

        updated_train_data = updated_training_data(existing_processed_data, processed_data)
        print(len(updated_train_data))
        updated_train_data += existing_train_data
        print(len(updated_train_data),len(existing_train_data))


        # Save updated data
        with open(args.train_json, 'w') as f:
            json.dump(updated_train_data, f, indent=2)
        # with open(args.val_json, 'w') as f:
        #     json.dump(val_data, f)
        print(f"Updated train data saved to {args.train_json}")
        # print(f"Updated validation data saved to {args.val_json}")

    print("Data processing completed!")


if __name__ == "__main__":
    main()