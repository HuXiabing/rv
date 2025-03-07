import os
import json
from pathlib import Path

def parse_throughput(line):
    """从倒数第二行提取 throughput 值"""
    if line.startswith("Cycle Min:"):
        return float(line.split("Cycle Min:")[1].strip())
    return None

def read_instructions(file_path):
    """读取 block 文件中的指令内容"""
    with open(file_path, "r") as f:
        lines = f.readlines()
    instructions = [line.strip() for line in lines if line.strip()]
    return instructions

def process_directory(directory):
    """处理指定目录下的所有文件"""
    results = []
    base_path = Path("random_generate/asm/")

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if not os.path.isfile(file_path):
            continue

        # 读取文件并提取 throughput
        with open(file_path, "r") as f:
            lines = f.readlines()
        if len(lines) < 2:
            continue

        throughput = parse_throughput(lines[-2].strip())
        if throughput is None:
            continue

        # 解析文件名，找到对应的 block 文件
        # 文件名格式：binary_test3.S.txt -> test3.S
        if not file_name.startswith("binary_") or not file_name.endswith(".txt"):
            continue
        block_file_name = file_name[len("binary_"):-len(".txt")]
        block_file_path = base_path / block_file_name

        if not block_file_path.exists():
            print(f"Warning: Block file {block_file_path} not found.")
            continue

        # 读取 block 文件中的指令
        instructions = read_instructions(block_file_path)

        # 添加到结果中
        results.append({
            "instructions": instructions,
            "throughput": throughput
        })

    return results

def save_to_json(data, output_file):
    """将结果保存为 JSON 文件"""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

def main():
    # 指定输入目录和输出文件
    input_directory = "random_generate/random_result"  # 替换为你的输入目录
    output_file = "data/output.json"  # 输出文件名

    # 处理目录并保存结果
    results = process_directory(input_directory)
    save_to_json(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
