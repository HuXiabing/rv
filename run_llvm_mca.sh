#!/bin/bash

# 默认值 ./run_llvm_mca.sh [-i my_asm_folder] [-o my_output_folder]
DEFAULT_ASM_DIR="random_generate/asm"
DEFAULT_OUTPUT_DIR="random_generate/mca"

rm -rf $DEFAULT_OUTPUT_DIR
mkdir -p $DEFAULT_OUTPUT_DIR

# 解析命令行参数
while getopts "i:o:" opt; do
  case $opt in
    i) ASM_DIR="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    *) echo "Usage: $0 [-i input_dir] [-o output_dir]"; exit 1 ;;
  esac
done

# 如果没有提供输入文件夹，则使用默认值
ASM_DIR="${ASM_DIR:-$DEFAULT_ASM_DIR}"
# 如果没有提供输出文件夹，则使用默认值
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

# 确保输出文件夹存在
mkdir -p "$OUTPUT_DIR"

# 遍历输入文件夹下的所有 .S 文件
for file in "$ASM_DIR"/*.S; do
    # 获取文件名（不带路径）
    filename=$(basename "$file" .S)
    # 定义输出文件路径
    output_file="$OUTPUT_DIR/$filename.S.txt"
    # 执行 llvm-mca 命令，并将结果重定向到输出文件
    /mnt/d/riscv/bin/llvm-mca -mcpu=xiangshan-nanhu -iterations=1000 "$file" > "$output_file"
    # 打印处理信息
    echo "Processed $file -> $output_file"
done

echo "All files processed. Input directory: $ASM_DIR, Output directory: $OUTPUT_DIR"