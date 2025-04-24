# #
# # from data.tokenizer import RISCVTokenizer
# # bb=["ld   a5,16(s0)", "ld    a0,0(s9)", "mv   a2,zero", "auipc  a1,21", "addi  a1,a1,576"]
# # tokenizer = RISCVTokenizer(max_instr_length=8)
# # token = []
# # encoded = []
# # for i in bb:
# #     encoded.append(tokenizer.encode_instruction(i))
# #     token.append(tokenizer.tokenize_instruction(i))
# # print(encoded)
# # print(token)

#
# import argparse
# import json
# import os
# from data.tokenizer import RISCVTokenizer
# from tqdm import tqdm
# import random
#
# def generate_json(cycle_jsons):
#     """Generate JSON data from ASM and cycle directories 合并所有json文件"""
#     all_results = {}
#     print(cycle_jsons)
#     for cycle_json in cycle_jsons:
#         with open("../bb/polybench/"+cycle_json, 'r') as f:
#             print(cycle_json)
#             data = json.load(f)
#         all_results.update(data)
#
#     # all_results = []
#     # for cycle_json in cycle_jsons:
#     #     with open(cycle_json, 'r') as f:
#     #         data = json.load(f)
#     #     all_results.extend(data)
#     return all_results
#
# def process_data(raw_data, tokenizer):
#     """Process raw data to add tokenization and encoding"""
#     processed_data = []
#
#     for item in tqdm(raw_data, desc="Processing data"):
#         # print("ien", item)
#         instructions = raw_data[item]["asm"].split("\n")
#
#         tokenized_instructions = []
#         for instr in instructions:
#             if "\\t" in instr:
#                 instr = instr.replace("\\t", "\t")
#             tokenized = tokenizer.tokenize_instruction(instr)
#             tokenized_instructions.append(tokenized)
#
#         valid = 1
#         encoded_instructions = []
#         for tokenized in tokenized_instructions:
#             encoded = [tokenizer.vocab.get(token, tokenizer.vocab.get('<PAD>', 0)) for token in tokenized]
#             if encoded[0] in list(range(73,137 + 1)):
#                 encoded_instructions.append(encoded)
#             else:
#                 valid = 0
#                 print(f"Warning: Invalid instruction: {tokenized}")
#
#         if valid == 0:
#             continue
#
#         # Create a processed sample
#         processed_item = {
#             "asm": raw_data[item]["asm"],
#             "binary": raw_data[item]["binary"].replace("\n", " "),
#             # "tokenized": tokenized_instructions,
#             "encoded": encoded_instructions,
#             # "num_instructions": len(instructions)
#         }
#
#         processed_data.append(processed_item)
#
#     return processed_data
#
# def get_encoded_key(item):
#     """Convert encoded instructions to a string for hashing"""
#     # Convert to JSON string for consistent representation
#     # print(item["encoded"])
#     return json.dumps(item["encoded"])
#
# def deduplicate_data(data):
#     """Deduplicate data based on the 'encoded' field"""
#     seen = set()
#     deduplicated = []
#
#     for item in data:
#         # print(item)
#         key = get_encoded_key(item)
#
#         if key not in seen:
#             seen.add(key)
#             item.pop("encoded", None)
#             deduplicated.append(item)
#     deduplicated_copy = deduplicated.copy()
#     random.seed(71)
#     random.shuffle(deduplicated_copy)
#     print(f"Deduplicated from {len(data)} to {len(deduplicated)} samples")
#     return deduplicated_copy
#
# def main():
#     print("--" * 20)
#     parser = argparse.ArgumentParser(description="RISC-V Instruction Throughput Data Processing")
#
#     parser.add_argument("--cycle_jsons", nargs="+", required=True,
#                         help="List of jsons containing cycle measurement files")
#
#     # Output paths
#     # parser.add_argument("--deduplicated_json", type=str, default="data/deduplicated_data.json",
#     #                     help="Output path for deduplicated JSON data")
#
#     # Output directory
#     parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
#
#     # Others
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#
#     args = parser.parse_args()
#
#
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     # Initialize tokenizer
#     tokenizer = RISCVTokenizer()
#
#     # Generate raw JSON data
#     print("Generating raw JSON data...")
#     # raw_data = generate_json(args.asm_dirs, args.cycle_dirs)
#     raw_data = generate_json(args.cycle_jsons)
#
#     # Process data - tokenize and encode
#     print("Processing data...")
#     processed_data = process_data(raw_data, tokenizer)
#
#     # Deduplicate processed data
#     processed_data = deduplicate_data(processed_data)
#     # processed_data = deduplicate_data(raw_data)
# if __name__ == "__main__":
#     main()


    # with open(args.deduplicated_json, 'w') as f:
    #     json.dump(processed_data, f, indent=2)
    # print(f"Deduplicated data saved to {args.deduplicated_json}, total samples: {len(processed_data)}")
# #-----------------
#     # filename = f"../bb/deduplicated_2w.json"
#     # with open(filename, 'w') as f:
#     #     json.dump(processed_data[:20000], f, indent=2)
#     chunk_size = 1500
#     total_chunks = (len(processed_data[:30000]) + chunk_size - 1) // chunk_size
#     for i in range(total_chunks):
#         start_idx = i * chunk_size
#         end_idx = min((i + 1) * chunk_size, len(processed_data))  # Ensure we don't go beyond the list length
#
#         # Create filename with chunk number
#         filename = f"../bb/deduplicated_{i + 1}.json"
#
#         # Write chunk to file
#         with open(filename, 'w') as f:
#             json.dump(processed_data[start_idx:end_idx], f, indent=2)
#
#         print(f"Saved chunk {i + 1}/{total_chunks}: items {start_idx} to {end_idx}")
#
#
# if __name__ == "__main__":
#     main()

# import json
# with open("data/exp1/train_data1.6w.json") as f:
#     train_data1 = json.load(f)
# with open("data/exp1/val_data.json") as f:
#     val_data = json.load(f)
# data = train_data1 + val_data
# print(len(data))
# import random
# random.seed(710)
# random.shuffle(data)
# train = data[:16000]
# val = data[16000:]
# with open("data/exp1/train_data1.6w_seed710.json", "w") as f:
#     json.dump(train, f, indent=2)
# with open("data/exp1/val_data_seed710.json", "w") as f:
#     json.dump(val, f, indent=2)
#
# with open("data/exp2/train_data8k_seed710.json", "w") as f:
#     json.dump(data[:8000], f, indent=2)
# with open("data/exp2/val_data_seed710.json", "w") as f:
#     json.dump(val, f, indent=2)
#
# with open("data/exp4/train_data4k_seed710.json", "w") as f:
#     json.dump(data[:4000], f, indent=2)
# with open("data/exp4/val_data_seed710.json", "w") as f:
#     json.dump(val, f, indent=2)


# import subprocess
# import tempfile
# import os
# import binascii
#
#
# def riscv_asm_to_hex(assembly_code):
#     # 创建临时文件保存汇编代码
#     with tempfile.NamedTemporaryFile(suffix='.s', delete=False) as asm_file:
#         asm_file.write(assembly_code.encode())
#         asm_file_name = asm_file.name
#
#     # 创建临时文件名用于目标文件
#     obj_file_name = asm_file_name + '.o'
#
#     try:
#         # 使用riscv64-unknown-linux-gnu-as汇编器将汇编代码编译为目标文件
#         subprocess.run(['riscv64-unknown-linux-gnu-as', '-march=rv64g', asm_file_name, '-o', obj_file_name], check=True, stderr=subprocess.DEVNULL)
#
#         # 使用riscv64-unknown-linux-gnu-objdump查看目标文件的十六进制内容
#         result = subprocess.run(['riscv64-unknown-linux-gnu-objdump', '-d', obj_file_name],
#                                 capture_output=True, text=True, check=True)
#
#         # 提取十六进制代码
#         hex_codes = []
#         # print(result.stdout.splitlines())
#         for line in result.stdout.splitlines():
#             if ':' in line:
#                 parts = line.split('\t')
#                 if len(parts) > 1:
#                     hex_part = parts[1].strip()
#                     if hex_part:
#                         hex_codes.append(hex_part)
#
#         return " ".join(hex_codes)
#
#     except subprocess.CalledProcessError as e:
#         print(f"Error during compilation: {e}")
#         return None
#     finally:
#         # 清理临时文件
#         if os.path.exists(asm_file_name):
#             os.remove(asm_file_name)
#         if os.path.exists(obj_file_name):
#             os.remove(obj_file_name)
#
#
# # 示例使用
# if __name__ == "__main__":
#     assembly_code = "addi\ts5,t3,0\nld\tt3,48(sp)\nslli\ta0,s6,0x20\naddi\ts3,t2,0\nslli\ta5,t3,0x2\naddi\ta3,a5,-4\naddi\tt2,s0,0\nsrli\ta2,a0,0x1e\nld\ts1,40(sp)\nld\ts0,32(sp)\nsub\ta3,a3,a2\naddi\tt4,a7,0\nadd\ta5,a4,a5\naddi\ta7,s8,0\nadd\ta3,a4,a3"
#
#     hex_codes = riscv_asm_to_hex(assembly_code)
#     print(hex_codes)

# import matplotlib.pyplot as plt
#
# 给定的数据
import numpy as np
# block_length_avg_loss_sorted = {
#         "2": 0.15303417899972258,
#         "3": 0.1435010446986096,
#         "4": 0.11000822113138131,
#         "5": 0.12596137068924446,
#         "6": 0.11319028162981147,
#         "7": 0.11201032446708843,
#         "8": 0.10406974003373301,
#         "9": 0.10503870146488183,
#         "10": 0.08595575019578627,
#         "11": 0.09522474362769709,
#         "12": 0.08545064871960724,
#         "13": 0.10248084917010244,
#         "14": 0.12185581933972273,
#         "15": 0.08123358270885157,
#         "16": 0.08119359142097467,
#         "17": 0.08291956350919516,
#         "18": 0.0860905953796228,
#         "19": 0.08773704915275347,
#         "20": 0.11432804220489093,
#         "21": 0.06734088268554346,
#         "22": 0.06283990145311691,
#         "23": 0.07353614615824293,
#         "24": 0.04943863382590387,
#         "25": 0.07728445297010088,
#         "26": 0.05155717742163688,
#         "27": 0.07233318160288035,
#         "28": 0.0697954622312234,
#         "29": 0.07145621324889362,
#         "30": 0.2902296530082822,
#         "31": 0.05639174673706293,
#         "32": 0.10694090388715267,
#         "33": 0.07535595785441274,
#         "34": 0.10626452881842852,
#         "35": 0.07757575921714306,
#         "36": 0.15849445015192032,
#         "37": 0.10471267501513164,
#         "38": 0.10308806660274665,
#         "39": 0.1658448005716006,
#         "40": 0.05259993311483413,
#         "41": 0.39780715107917786,
#         "42": 0.06232406198978424,
#         "43": 0.06951956947644551,
#         "44": 0.11482758820056915,
#         "45": 0.0017101058037951589,
#         "46": 0.1342196762561798,
#         "47": 0.1443121749907732,
#         "48": 0.15799706677595773,
#         "50": 0.09081015661358834,
#         "51": 0.11422442644834518,
#         "52": 0.05137116089463234,
#         "53": 0.05399308539927006,
#         "54": 0.025878584012389183,
#         "56": 0.11226561665534973,
#         "57": 0.3186263144016266,
#         "59": 0.12167630344629288,
#         "60": 0.22097505629062653,
#         "61": 0.11190925911068916,
#         "62": 0.07939277961850166,
#         "64": 0.04468459635972977,
#         "66": 0.38692423701286316,
#         "68": 0.08665996417403221,
#         "69": 0.0045000528916716576,
#         "70": 0.0346587598323822,
#         "76": 0.011232731398195028,
#         "79": 0.27425211668014526,
#         "83": 0.18098703026771545,
#         "91": 0.12210360914468765,
#         "92": 0.006156034767627716,
#         "94": 0.31641483306884766,
#         "99": 0.2383236140012741,
#         "102": 0.4121474325656891,
#         "111": 0.3460398018360138,
#         "117": 0.3697846829891205,
#         "121": 0.3614669144153595,
#         "125": 0.4273291230201721,
#         "137": 0.42773497104644775,
#         "184": 0.6306410431861877,
#         "387": 0.8645846247673035
#     }
# block_length_counts_sorted =  {
#         "2": 433,
#         "3": 577,
#         "4": 547,
#         "5": 454,
#         "6": 358,
#         "7": 247,
#         "8": 221,
#         "9": 193,
#         "10": 168,
#         "11": 106,
#         "12": 95,
#         "13": 79,
#         "14": 61,
#         "15": 56,
#         "16": 47,
#         "17": 34,
#         "18": 37,
#         "19": 29,
#         "20": 21,
#         "21": 22,
#         "22": 16,
#         "23": 17,
#         "24": 17,
#         "25": 15,
#         "26": 20,
#         "27": 10,
#         "28": 13,
#         "29": 8,
#         "30": 4,
#         "31": 5,
#         "32": 5,
#         "33": 2,
#         "34": 4,
#         "35": 5,
#         "36": 2,
#         "37": 3,
#         "38": 6,
#         "39": 3,
#         "40": 3,
#         "41": 1,
#         "42": 2,
#         "43": 3,
#         "44": 3,
#         "45": 1,
#         "46": 2,
#         "47": 2,
#         "48": 3,
#         "50": 5,
#         "51": 1,
#         "52": 1,
#         "53": 2,
#         "54": 1,
#         "56": 1,
#         "57": 1,
#         "59": 1,
#         "60": 1,
#         "61": 2,
#         "62": 2,
#         "64": 1,
#         "66": 1,
#         "68": 2,
#         "69": 1,
#         "70": 1,
#         "76": 2,
#         "79": 1,
#         "83": 1,
#         "91": 1,
#         "92": 1,
#         "94": 1,
#         "99": 1,
#         "102": 1,
#         "111": 1,
#         "117": 1,
#         "121": 1,
#         "125": 1,
#         "137": 1,
#         "184": 1,
#         "387": 1
#     }
#
# # 将键转换为整数并排序
# sorted_lengths = sorted(int(k) for k in block_length_avg_loss_sorted.keys())
# sorted_counts = [block_length_counts_sorted[str(k)] for k in sorted_lengths]
#
# sorted_loss = [block_length_avg_loss_sorted[str(k)] for k in sorted_lengths]
# result = [x / y for x, y in zip(sorted_loss, sorted_counts)]
# from scipy.stats import pearsonr
# # print(sorted_counts)
# # print(result)
# corr, p_value = pearsonr(sorted_counts[-50:], result[-50:])
#
# print(f"Pearson相关系数: {corr:.4f}")
# print(f"P值: {p_value:.4g}")

# # 创建图表
# plt.figure(figsize=(15, 6))
#
# # 绘制柱状图
# bars = plt.bar(sorted_lengths, result)
#
# # 添加标签和标题
# plt.xlabel('块长度', fontsize=12)
# plt.ylabel('出现次数', fontsize=12)
# plt.title('不同长度基本块的出现频率分布', fontsize=14)
#
# # 调整x轴刻度标签
# plt.xticks(sorted_lengths, rotation=90, fontsize=8)
#
# # 自动调整布局防止标签重叠
# plt.tight_layout()
#
# # 显示图表
# plt.show()

import numpy as np
from scipy.stats import pearsonr

# 原始数据
category_loss_sum = {
    "arithmetic": 0.12924855879462846,
    "shifts": 0.3248663967591554,
    "logical": 0.1897330593082105,
    "compare": 0.11614487947220464,
    "mul": 0.05692000195050402,
    "div": 0.7655404993938102,
    "rem": 0.44559135997579213,
    "load": 0.18272623450874625,
    "store": 0.055525469294281184
}

category_count_sum = {
    "arithmetic": 14534,
    "shifts": 1671,
    "logical": 502,
    "compare": 94,
    "mul": 471,
    "div": 16,
    "rem": 15,
    "load": 7895,
    "store": 6903
}

# 确保键的顺序一致
categories = category_loss_sum.keys()
losses = np.array([category_loss_sum[cat] for cat in categories])
counts = np.array([category_count_sum[cat] for cat in categories])
result = [x / y for x, y in zip(losses, counts)]
# 计算Pearson相关系数
print(categories)
print(counts)
print(result)
corr, p_value = pearsonr(counts[5:], result[5:])

print(f"Pearson相关系数: {corr:.4f}")
print(f"P值: {p_value:.4g}")