#!/bin/bash

# 检查输入参数
if [ $# -ne 2 ]; then
    echo "用法：$0 <输入目录> <输出目录>"
    exit 1
fi


# 获取绝对路径并创建输出目录
INPUT_DIR=$(realpath -m "$1")
OUTPUT_DIR=$(realpath -m "$2")
mkdir -p "$OUTPUT_DIR" || { echo "无法创建输出目录！"; exit 1; }

# 设置用于处理空格的字段分隔符
SAVE_IFS=$IFS
IFS=$'\n'

# 处理所有.S文件
find "$INPUT_DIR" -type f -name "*.S" -print0 | while IFS= read -r -d $'\0' S_FILE; do
    # 计算相对路径
    REL_PATH=$(realpath --relative-to="$INPUT_DIR" "$S_FILE")
    
    # 创建目标目录
    TARGET_DIR="$OUTPUT_DIR/$(dirname "$REL_PATH")"
    mkdir -p "$TARGET_DIR" || { echo "无法创建目录 $TARGET_DIR"; continue; }

    # 生成文件名
    BASE_NAME=$(basename "$S_FILE" .S)
    OBJ="$TARGET_DIR/${BASE_NAME}.o"
    ELF="$TARGET_DIR/${BASE_NAME}.elf"
    BIN="$TARGET_DIR/binary_${BASE_NAME}.S"

    echo "正在处理: $S_FILE"
    
    # 汇编阶段（添加调试信息）
    if ! riscv64-unknown-linux-gnu-as -g -o "$OBJ" "$S_FILE"; then
        echo "汇编失败: $S_FILE"
        rm -f "$OBJ"
        continue
    fi
    
    # 链接阶段（指定代码段地址）
    if ! riscv64-unknown-linux-gnu-ld -Ttext=0x00000000 -o "$ELF" "$OBJ"; then
        echo "链接失败: $S_FILE"
        rm -f "$OBJ" "$ELF"
        continue
    fi
    
    # 提取机器码（增强模式匹配）
    if riscv64-unknown-linux-gnu-objdump -d "$ELF" | \
       awk '/^ *[0-9a-f]+:\t/ { print $2 }' > "$BIN"; then
        echo "已生成: $BIN"
        # 验证输出文件
        if [ ! -s "$BIN" ]; then
            echo "警告: 生成的二进制文件为空 $BIN"
        fi
    else
        echo "提取失败: $S_FILE"
    fi
    
    # 清理中间文件
    rm -f "$OBJ" "$ELF"
done

IFS=$SAVE_IFS
echo "所有文件处理完成！输出目录: $OUTPUT_DIR"
