#!/bin/bash

# 设置目录路径
INPUT_DIR="./random_generate"
OUTPUT_DIR="./random_generate"

# 检查run_llvm_mca.sh是否存在且可执行
if [ ! -x "./run_llvm_mca.sh" ]; then
    echo "Error: run_llvm_mca.sh not found or not executable"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

# 创建一个数组来存储所有后台进程的PID
pids=()

# 并行启动所有处理命令
for i in {0..9}; do
    input_file="${INPUT_DIR}/asm${i}.json"
    output_file="${OUTPUT_DIR}/output${i}.json"
    
    # 检查输入文件是否存在
    if [ -f "$input_file" ]; then
        echo "Starting process: $input_file -> $output_file"
        
        # 在后台运行命令
        ./run_llvm_mca.sh "$input_file" "$output_file" &
        
        # 保存进程ID
        pids+=($!)
    else
        echo "Warning: Input file $input_file does not exist, skipping"
    fi
done

# 等待所有后台进程完成
echo "Waiting for all processes to complete..."
for pid in "${pids[@]}"; do
    wait $pid
    status=$?
    if [ $status -eq 0 ]; then
        echo "Process $pid completed successfully"
    else
        echo "Process $pid failed with status $status"
    fi
done

echo "All processing completed!"
