#!/bin/bash

# 使用方法: ./run_test.sh <输入目录> <输出目录>

# 检查参数数量
if [ $# -ne 2 ]; then
    echo "错误: 参数数量不正确"
    echo "用法: $0 <输入目录> <输出目录>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
TIMEOUT_SECONDS=10

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 '$INPUT_DIR' 不存在"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 检查输出目录是否创建成功
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "错误: 无法创建输出目录 '$OUTPUT_DIR'"
    exit 1
fi

# 检查test程序是否存在且可执行
if [ ! -x "./test" ]; then
    echo "错误: './test' 不存在或不可执行"
    echo "请确保在当前目录下有可执行的 'test' 程序"
    exit 1
fi

# 计数器
total_files=0
processed_files=0
success_count=0
timeout_count=0
failed_count=0

# 获取文件总数
for file in "$INPUT_DIR"/*; do
    if [ -f "$file" ]; then
        ((total_files++))
    fi
done

echo "开始处理 $total_files 个文件..."

# 超时函数 - 使用 Bash 内置的 SECONDS 变量实现超时控制
run_with_timeout() {
    local cmd="$1"
    local timeout=$2
    local output_file="$3"

    # 重置 SECONDS 计时器
    SECONDS=0

    # 启动命令到后台并获取其 PID
    eval "$cmd" > "$output_file" 2>&1 &
    local pid=$!

    # 每 0.1 秒检查一次进程是否完成或超时
    local step=0.1
    local elapsed=0

    while kill -0 $pid 2>/dev/null; do
        sleep $step
        elapsed=$(echo "$SECONDS" | awk '{printf "%.1f", $1}')

        # 检查是否超时
        if (( $(echo "$elapsed >= $timeout" | bc -l) )); then
            kill -9 $pid 2>/dev/null
            wait $pid 2>/dev/null
            echo "  错误: 执行超时(>${timeout}秒)，已终止" | tee -a "$output_file"
            return 124  # 使用与 timeout 命令相同的返回码
        fi
    done

    # 等待后台进程完成并获取退出码
    wait $pid
    return $?
}

# 休息和清理函数
rest_and_clean() {
    echo "运行时间已达 1 小时，开始休息和清理..."

    # 休息 1 分钟
    echo "休息 1 分钟..."
    sleep 60

    # 杀死所有 ./test 进程
    echo "杀死所有 ./test 进程..."
    ps aux | awk '/\.\/test/ {print $1}' | xargs kill -9 2>/dev/null

    # 再次休息 1 分钟
    echo "再次休息 1 分钟..."
    sleep 60

    # 重置计时器
    SECONDS=0
}

# 重置计时器
SECONDS=0

# 处理每个文件
for file in "$INPUT_DIR"/*; do
    # 检查是否为普通文件
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        output_file="$OUTPUT_DIR/${filename}.txt"

        # echo "处理 ($((processed_files+1))/$total_files): $filename"

        # 使用超时控制运行命令
        run_with_timeout "./test \"$file\"" $TIMEOUT_SECONDS "$output_file"
        exit_code=$?

        # 检查命令执行状态
        if [ $exit_code -eq 0 ]; then
           # echo "  成功: 输出已保存到 $output_file"
            ((success_count++))
        elif [ $exit_code -eq 124 ]; then  # 超时
            echo "  输出已保存到 $output_file"
            ((timeout_count++))
        else
            echo "  警告: 命令执行失败(退出码: $exit_code)，输出已保存到 $output_file"
            ((failed_count++))
        fi

        ((processed_files++))
    fi

    # 检查是否需要休息
    if (( SECONDS >= 3600 )); then
        rest_and_clean
    fi
done

echo "处理完成: 共处理 $processed_files 个文件"
echo "  成功: $success_count 个"
echo "  超时: $timeout_count 个"
echo "  失败: $failed_count 个"
exit 0
