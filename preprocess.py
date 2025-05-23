import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import random
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.tokenizer import RISCVTokenizer

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

def generate_json_mca(cycle_jsons,shuffle=True):
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

    if shuffle == False:
        return all_results
    else:
        all_results_copy = all_results.copy()
        random.seed(71)
        random.shuffle(all_results_copy)
    return all_results_copy

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

            all_results.append({
                "instructions": instructions.split("\n"),
                "throughput": throughput})
    return all_results

def process_data(raw_data, tokenizer, id=False): # 4516
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
            # print(tokenized)
            encoded = [tokenizer.vocab.get(token, tokenizer.vocab.get('<PAD>', 0)) for token in tokenized]
            if encoded[0] in list(range(73,137 + 1)):
                encoded_instructions.append(encoded)
            else:
                valid = 0
                print(f"Warning: Invalid instruction: {tokenized}")

        if valid == 0:
            continue

        if id == True:
            item["idx"] = idx

        # Create a processed sample
        processed_item = {
            "instructions": instructions,
            "tokenized": tokenized_instructions,
            "encoded": encoded_instructions,
            "throughput": throughput,
            "idx": item.get("idx",None),
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

    for idx, item in enumerate(tqdm(data, desc="Deduplicating data")):
        key = get_encoded_key(item)
        if key not in seen:
            seen.add(key)
            item.pop("tokenized",None)
            item.pop("encoded",None)
            idx = item.get('idx', idx)
            deduplicated.append(item)


    print(f"Deduplicated from {len(data)} to {len(deduplicated)} samples")
    return deduplicated

def updated_training_data(existing_train_data, existing_val_data, processed_data):
    """Update training data with new data while keeping validation set unchanged"""
    existing_processed_data = existing_train_data + existing_val_data
    processed_data_encoded_keys = {get_encoded_key(item) for item in existing_processed_data}
    idx = len(existing_train_data)
    filtered_new_data = []
    for item in processed_data:
        key = get_encoded_key(item)
        if key not in processed_data_encoded_keys:
            processed_data_encoded_keys.add(key)
            item.pop("tokenized", None)
            item.pop("encoded", None)
            item['idx'] = idx
            idx += 1
            filtered_new_data.append(item)
    print(f"Added {len(filtered_new_data)} new samples to training data")
    return filtered_new_data

def main():
    parser = argparse.ArgumentParser(description="RISC-V Instruction Throughput Data Processing")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["full", "incremental"], default="incremental",
                        help="Processing mode: full (process from scratch) or incremental (add new data)")
    parser.add_argument("--cycle_jsons", nargs="+", required=True,
                        help="List of jsons containing cycle measurement files")
    # Existing data for incremental mode
    parser.add_argument("--existing_train_json", type=str, default="data/train_data.json",
                        help="Path to the existing train JSON file (for incremental mode)")
    parser.add_argument("--existing_val_json", type=str, default="data/val_data.json",
                        help="Path to the existing validation JSON file (for incremental mode)")
    parser.add_argument("--train_json", type=str, default="data/train_data.json",
                        help="Output path for training JSON data")
    parser.add_argument("--val_json", type=str, default="data/val_data.json",
                        help="Output path for validation JSON data")
    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = RISCVTokenizer()
    print("Generating raw JSON data...")
    raw_data = generate_json_mca(args.cycle_jsons, shuffle=True)

    if args.mode == "full":

        # Process data - tokenize and encode
        print("Processing data...")
        processed_data = process_data(raw_data, tokenizer, id = True)
        """
        processed_item = {
                "instructions": instructions,
                "tokenized": tokenized_instructions,
                "encoded": encoded_instructions,
                "throughput": throughput,
                "idx": idx,
            }
        """

        processed_data = deduplicate_data(processed_data)
        print("saved processed data: ", len(processed_data))

        # Split data into train and validation sets
        print(len(processed_data))
        with open("random_generate/xiangshan.json", 'w') as f:
            json.dump(processed_data, f, indent=2)
        # Save split data to JSON
        with open("random_generate/xiangshan/train_data1.json", 'w') as f:
            json.dump(processed_data[:10000], f, indent=2)
        with open("random_generate/xiangshan/train_data2.json", 'w') as f:
            json.dump(processed_data[:20000], f, indent=2)
        with open("random_generate/xiangshan/train_data5.json", 'w') as f:
            json.dump(processed_data[:50000], f, indent=2)
        with open("random_generate/xiangshan/train_data10.json", 'w') as f:
            json.dump(processed_data[:100000], f, indent=2)
        with open("random_generate/xiangshan/train_data20.json", 'w') as f:
            json.dump(processed_data[:200000], f, indent=2)
        with open("random_generate/xiangshan/train_data30.json", 'w') as f:
            json.dump(processed_data[:300000], f, indent=2)
        with open("random_generate/xiangshan/train_data40.json", 'w') as f:
            json.dump(processed_data[:400000], f, indent=2)
        with open("random_generate/xiangshan/train_data50.json", 'w') as f:
            json.dump(processed_data[:500000], f, indent=2)
        with open("random_generate/xiangshan/train_data60.json", 'w') as f:
            json.dump(processed_data[:600000], f, indent=2)
        with open("random_generate/xiangshan/train_data70.json", 'w') as f:
            json.dump(processed_data[:700000], f, indent=2)
        with open("random_generate/xiangshan/train_data80.json", 'w') as f:
            json.dump(processed_data[:800000], f, indent=2)
        with open("random_generate/xiangshan/val_data.json", 'w') as f:
            json.dump(processed_data[800000:], f, indent=2)
        # with open("data/xiangshan/val_data.json", 'w') as f:
        #     json.dump(processed_data[300:], f, indent=2)

    else:
        print("Generating raw JSON data...")
        raw_data = generate_json_mca(args.cycle_jsons, shuffle=True)
        """
        {"instructions": instructions.split("\n"),
         "throughput": throughput}
        """

        print("Loading existing processed data...")
        with open(args.existing_train_json, 'r') as f:
            existing_train_data = json.load(f)
        with open(args.existing_val_json, 'r') as f:
            existing_val_data = json.load(f)

        # Process data - tokenize and encode
        print("Processing data...")
        processed_train_data = process_data(existing_train_data, tokenizer)
        processed_val_data = process_data(existing_val_data, tokenizer)
        processed_data = process_data(raw_data, tokenizer)
        """
        processed_item = {
                "instructions": instructions,
                "tokenized": tokenized_instructions,
                "encoded": encoded_instructions,
                "throughput": throughput,
                "idx": idx,
            }
        """

        updated_train_data = updated_training_data(processed_train_data, processed_val_data, processed_data)
        train_data = existing_train_data + updated_train_data

        # Save updated data
        with open(args.train_json, 'w') as f:
            json.dump(train_data, f, indent=2)

        print(f"Updated train data saved to {args.train_json}")
    print("Data processing completed!")


if __name__ == "__main__":
    main()