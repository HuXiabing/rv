import pprint as pp
import sys
import json
from rvmca.gen import gen_block
from rvmca.gen.inst_gen import gen_block_vector, DependencyAnalyzer
import argparse
import os

import numpy as np
import random
from typing import Dict, List

import numpy as np
import random
from typing import Dict, List, Tuple

import numpy as np
import random
from typing import Dict, List, Tuple


class EnhancedFuzzer:
    """
    增强型智能Fuzzer，利用loss反馈机制定向生成测试基本块

    通过学习不同指令类型和基本块长度的loss分布，智能调整生成策略
    同时保持一定随机性以探索更广泛的测试空间
    """

    def __init__(self,
                 instruction_avg_loss: Dict[str, float],
                 instruction_counts: Dict[str, int],
                 block_length_avg_loss: Dict[str, float],
                 block_length_counts: Dict[str, int],
                 temp: float = 0.5,
                 explore_rate: float = 0.2):
        """
        初始化Fuzzer

        Args:
            instruction_avg_loss: 各指令类型的平均loss字典
            instruction_counts: 各指令类型的出现次数字典
            block_length_avg_loss: 各基本块长度的平均loss字典
            block_length_counts: 各基本块长度的出现次数字典
            temp: 温度系数，控制loss影响的强度
            explore_rate: 探索率，控制随机探索的比例
        """
        # 指令分类映射表
        self.instr_categories = {
            # 移位和算术运算
            'add': 'shifts_arithmetic', 'addi': 'shifts_arithmetic',
            'sub': 'shifts_arithmetic', 'sll': 'shifts_arithmetic',
            'slli': 'shifts_arithmetic', 'srl': 'shifts_arithmetic',
            'srli': 'shifts_arithmetic', 'sra': 'shifts_arithmetic',
            'srai': 'shifts_arithmetic', 'or': 'shifts_arithmetic',
            'ori': 'shifts_arithmetic', 'and': 'shifts_arithmetic',
            'andi': 'shifts_arithmetic', 'xor': 'shifts_arithmetic',
            'xori': 'shifts_arithmetic',

            # 比较指令
            'slt': 'compare', 'slti': 'compare', 'sltu': 'compare',
            'sltiu': 'compare', 'beq': 'compare', 'bne': 'compare',
            'blt': 'compare', 'bge': 'compare', 'bltu': 'compare',
            'bgeu': 'compare',

            # 乘除法
            'mul': 'mul_div', 'mulh': 'mul_div', 'mulhu': 'mul_div',
            'mulhsu': 'mul_div', 'div': 'mul_div', 'divu': 'mul_div',
            'rem': 'mul_div', 'remu': 'mul_div',

            # 加载指令
            'lb': 'load', 'lh': 'load', 'lw': 'load', 'ld': 'load',
            'lbu': 'load', 'lhu': 'load', 'lwu': 'load',

            # 存储指令
            'sb': 'store', 'sh': 'store', 'sw': 'store', 'sd': 'store'
        }

        # 指令类型配置
        self.type_order = ['shifts_arithmetic', 'compare', 'mul_div', 'load', 'store']
        self.type_weights = self._aggregate_instruction_loss(instruction_avg_loss, instruction_counts)

        # 长度策略配置
        self.length_stats = self._process_length_stats(block_length_avg_loss, block_length_counts)
        self.length_probs = self._build_length_distribution(self.length_stats)
        self.length_clusters = self._identify_length_clusters(self.length_stats)

        # 控制参数
        self.temp = temp  # 温度系数
        self.explore_rate = explore_rate  # 探索率
        self.depen_boost = [0.5, 0.5, 0.5]  # 依赖关系增强因子 [WAW, RAW, WAR]

        # 性能追踪
        self.perf_history = []  # 记录平均loss历史
        self.update_counter = 0  # 更新计数器

        # 保存原始数据
        self.instruction_loss_map = instruction_avg_loss.copy()
        self.instruction_count_map = instruction_counts.copy()

    def _aggregate_instruction_loss(self,
                                    instr_loss: Dict[str, float],
                                    instr_counts: Dict[str, int]) -> np.ndarray:
        """
        将指令级别的loss聚合到类别级别

        Args:
            instr_loss: 指令级别的loss字典
            instr_counts: 指令级别的计数字典

        Returns:
            类别级别的loss权重数组
        """
        # 初始化类别loss累加器
        category_loss_sum = {cat: 0.0 for cat in self.type_order}
        category_count_sum = {cat: 0 for cat in self.type_order}

        # 按类别聚合loss和计数
        for instr, loss in instr_loss.items():
            category = self.instr_categories.get(instr)
            if category and category in self.type_order:
                count = instr_counts.get(instr, 1)  # 如果没有计数数据，默认为1
                category_loss_sum[category] += loss * count
                category_count_sum[category] += count

        # 计算每个类别的加权平均loss
        category_weights = []
        for cat in self.type_order:
            if category_count_sum[cat] > 0:
                avg_loss = category_loss_sum[cat] / category_count_sum[cat]
            else:
                avg_loss = 0.1  # 对于没有数据的类别，使用默认值
            category_weights.append(avg_loss)

        # 转换为numpy数组并标准化
        weights = np.array(category_weights)
        return weights / max(weights.sum(), 1e-6)

    def _process_length_stats(self,
                              length_loss: Dict[str, float],
                              length_counts: Dict[str, int]) -> Dict[int, Dict]:
        """
        处理长度统计数据为更易用的格式

        Args:
            length_loss: 长度到loss的映射字典
            length_counts: 长度到计数的映射字典

        Returns:
            长度到统计信息的映射字典
        """
        length_stats = {}

        # 合并两个字典的键集合
        all_lengths = set(length_loss.keys()) | set(length_counts.keys())

        for l_str in all_lengths:
            length = int(l_str)
            count = length_counts.get(l_str, 0)
            loss = length_loss.get(l_str, 0.1)  # 默认loss值

            # 根据样本数量计算置信度
            confidence = min(1.0, count / 100)  # 样本数超过100时置信度为1.0

            # 存储长度统计信息
            length_stats[length] = {
                'loss': loss,
                'count': count,
                'confidence': confidence,
                'weighted_loss': loss * (0.3 + 0.7 * confidence)  # 加权loss
            }

        return length_stats

    def _build_length_distribution(self, length_stats: Dict[int, Dict]) -> Dict[int, float]:
        """
        构建长度概率分布

        Args:
            length_stats: 长度统计信息

        Returns:
            长度到选择概率的映射
        """
        if not length_stats:
            # 默认均匀分布
            default_lengths = [4, 8, 12, 16, 20]
            return {l: 1.0 / len(default_lengths) for l in default_lengths}

        # 提取加权loss并计算softmax分布
        lengths = list(length_stats.keys())
        weighted_losses = [stats['weighted_loss'] for stats in length_stats.values()]

        # 应用温度缩放的softmax
        exp_losses = np.exp(np.array(weighted_losses) / self.temp)
        probs = exp_losses / exp_losses.sum()

        return {length: prob for length, prob in zip(lengths, probs)}

    def _identify_length_clusters(self, length_stats: Dict[int, Dict]) -> List[Tuple[int, float]]:
        """
        识别有前景的长度聚类

        Args:
            length_stats: 长度统计信息

        Returns:
            长度聚类列表，每个元素为(中心点, 扩散范围)
        """
        if not length_stats:
            return [(10, 3.0)]  # 默认聚类

        # 按加权loss降序排序
        sorted_lengths = sorted(
            [(length, stats['weighted_loss']) for length, stats in length_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # 取loss排名前3的长度作为聚类中心（或者更少如果长度不足）
        top_n = min(3, len(sorted_lengths))
        clusters = []

        for i in range(top_n):
            center = sorted_lengths[i][0]

            # 确定扩散范围（基于附近长度的可用性）
            nearby_lengths = [l for l in length_stats.keys() if abs(l - center) <= 5]
            if len(nearby_lengths) > 1:
                spread = max(2.0, np.std(nearby_lengths))
            else:
                spread = 3.0  # 默认扩散范围

            clusters.append((center, spread))

        return clusters

    def _adapt_type_ratios(self) -> List[float]:
        """
        根据当前策略生成指令类型比例

        Returns:
            各指令类型的比例列表
        """
        # 应用温度缩放
        scaled_weights = np.exp(self.type_weights / self.temp)
        base_ratios = scaled_weights / scaled_weights.sum()

        # 如果处于探索模式，添加随机噪声
        if random.random() < self.explore_rate:
            noise = np.random.uniform(-0.2, 0.2, len(base_ratios))
            # 确保不会产生负值或过高的值
            ratios = np.clip(base_ratios + noise, 0.05, 0.5)
            # 重新归一化
            ratios = ratios / ratios.sum()
        else:
            ratios = base_ratios

        return ratios.tolist()

    def _choose_length(self) -> int:
        """
        智能选择基本块长度

        Returns:
            选择的长度值
        """
        # 探索与利用的策略选择
        if random.random() < self.explore_rate:
            # 探索模式：尝试更广泛的长度范围
            if random.random() < 0.7 and self.length_clusters:
                # 从随机选择的聚类中采样，但更大的变异
                cluster_idx = random.randint(0, len(self.length_clusters) - 1)
                center, spread = self.length_clusters[cluster_idx]
                return max(1, int(np.random.normal(center, spread * 1.5)))
            else:
                # 有时尝试完全新的长度（使用长尾分布）
                return max(1, min(50, int(np.random.lognormal(1.5, 0.6))))
        else:
            # 利用模式：使用学习到的分布
            if self.length_probs and random.random() < 0.8:
                # 直接从分布中采样
                lengths = list(self.length_probs.keys())
                probs = list(self.length_probs.values())
                return np.random.choice(lengths, p=probs)
            else:
                # 从聚类中采样
                if self.length_clusters:
                    # 从随机选择的聚类中采样
                    cluster_idx = random.randint(0, len(self.length_clusters) - 1)
                    center, spread = self.length_clusters[cluster_idx]
                    return max(1, int(np.random.normal(center, spread)))
                else:
                    # 使用默认值
                    return random.choice([4, 8, 12, 16])

    def _gen_dependency_flags(self) -> List[int]:
        """
        生成数据依赖关系标志

        Returns:
            依赖关系标志列表 [WAW依赖, RAW依赖, WAR依赖]
            WAW (Write After Write): 相同寄存器的两次写入
            RAW (Read After Write): 读取之前写入的寄存器
            WAR (Write After Read): 写入之前读取的寄存器
        """
        # 获取各类型指令权重
        arith_weight = self.type_weights[0]  # 算术/移位/逻辑指令权重
        compare_weight = self.type_weights[1]  # 比较指令权重
        mul_div_weight = self.type_weights[2]  # 乘除法指令权重
        load_weight = self.type_weights[3]  # 加载指令权重
        store_weight = self.type_weights[4]  # 存储指令权重

        # 依赖概率计算 - 基于指令类型特性
        # WAW: 写入相同寄存器，与写入寄存器的指令相关 (算术/存储指令更常有def寄存器)
        waw_dep_prob = self.depen_boost[0] + 0.3 * (arith_weight + store_weight)

        # RAW: 读取之前写入的值，多数指令都可能读取值 (加载/算术对读取值依赖性更强)
        raw_dep_prob = self.depen_boost[1] + 0.3 * (load_weight + arith_weight + mul_div_weight)

        # WAR: 写入之前读取的寄存器，多用于寄存器重用
        war_dep_prob = self.depen_boost[2] + 0.2 * (arith_weight + compare_weight)

        # 限制依赖概率在合理范围内
        probs = [
            np.clip(waw_dep_prob, 0.1, 0.9),
            np.clip(raw_dep_prob, 0.1, 0.9),
            np.clip(war_dep_prob, 0.1, 0.9)
        ]

        # 生成依赖标志
        return [int(random.random() < p) for p in probs]

    def _calculate_depth(self, length: int) -> int:
        """
        计算适当的深度参数

        Args:
            length: 基本块长度

        Returns:
            深度参数
        """
        # 增强的深度计算逻辑
        if length <= 3:
            return 1
        elif length <= 7:
            return 2
        elif length <= 12:
            return 3
        elif length <= 20:
            return 4
        else:
            return 5

    def update_strategy(self,
                        instruction_avg_loss: Dict[str, float],
                        block_length_avg_loss: Dict[str, float],
                        avg_loss: float = None):
        """
        根据新的loss信息更新策略

        Args:
            instruction_avg_loss: 新的指令类型loss字典
            block_length_avg_loss: 新的基本块长度loss字典
            avg_loss: 可选的整体平均loss
        """
        self.update_counter += 1

        # 更新指令类型权重
        new_weights = self._aggregate_instruction_loss(
            instruction_avg_loss, self.instruction_count_map)

        # 自适应学习率
        lr = 0.3  # 基础学习率

        # 如果提供了平均loss，追踪并调整学习率
        if avg_loss is not None:
            self.perf_history.append(avg_loss)

            # 如果有足够的历史数据，调整学习率
            if len(self.perf_history) >= 5:
                recent_losses = self.perf_history[-5:]
                trend = recent_losses[-1] / max(np.mean(recent_losses[:-1]), 1e-5) - 1

                # 根据趋势调整学习率
                if trend > 0.05:  # 性能明显改善
                    lr = min(0.5, lr * 1.2)
                elif trend < -0.05:  # 性能明显下降
                    lr = max(0.1, lr * 0.8)

        # 应用学习率更新类型权重
        self.type_weights = (1 - lr) * self.type_weights + lr * new_weights
        self.type_weights /= self.type_weights.sum()  # 重新归一化

        # 更新长度统计和分布
        new_length_stats = self._process_length_stats(
            block_length_avg_loss, self.length_stats)

        # 更新现有长度的统计信息
        for length, new_stats in new_length_stats.items():
            if length in self.length_stats:
                old_stats = self.length_stats[length]

                # 根据loss变化大小调整学习率
                change_magnitude = abs(new_stats['loss'] - old_stats['loss']) / max(old_stats['loss'], 1e-5)
                adaptive_lr = min(0.5, lr * (1 + change_magnitude))

                # 更新loss和加权loss
                old_stats['loss'] = (1 - adaptive_lr) * old_stats['loss'] + adaptive_lr * new_stats['loss']
                old_stats['count'] += 1
                old_stats['confidence'] = min(1.0, old_stats['count'] / 100)
                old_stats['weighted_loss'] = old_stats['loss'] * (0.3 + 0.7 * old_stats['confidence'])
            else:
                # 新的长度，直接添加
                self.length_stats[length] = new_stats

        # 更新派生模型
        self.length_probs = self._build_length_distribution(self.length_stats)
        self.length_clusters = self._identify_length_clusters(self.length_stats)

        # 定期更新元参数
        if self.update_counter % 10 == 0 and len(self.perf_history) >= 10:
            self._update_meta_parameters()

    def _update_meta_parameters(self):
        """根据性能历史更新温度和探索率"""
        if len(self.perf_history) < 10:
            return  # 数据不足

        # 获取最近的loss历史
        history = self.perf_history
        recent = history[-10:]

        # 检查性能趋势
        is_improving = recent[-1] > np.mean(recent[:5])
        is_stagnating = abs(recent[-1] - np.mean(recent[:5])) / max(np.mean(recent[:5]), 1e-5) < 0.03

        # 调整温度
        if is_improving:
            # 如果性能改善，逐渐降低温度
            self.temp = max(0.3, self.temp * 0.95)
        elif is_stagnating:
            # 如果性能停滞，更积极地增加温度
            self.temp = min(1.0, self.temp * 1.15)
        else:
            # 如果性能变差，略微增加温度
            self.temp = min(0.8, self.temp * 1.05)

        # 调整探索率
        if is_improving:
            # 如果性能改善，逐渐降低探索率
            self.explore_rate = max(0.1, self.explore_rate * 0.9)
        elif is_stagnating:
            # 如果性能停滞，更积极地增加探索率
            self.explore_rate = min(0.4, self.explore_rate * 1.2)
        else:
            # 如果性能变差，略微增加探索率
            self.explore_rate = min(0.3, self.explore_rate * 1.1)

    def generate(self) -> List:
        """
        生成测试基本块

        Returns:
            生成的基本块向量表示
        """
        # 自适应选择参数
        ratios = self._adapt_type_ratios()
        length = self._choose_length()
        dep_flags = self._gen_dependency_flags()
        depth = self._calculate_depth(length)

        # 生成基本块向量
        return gen_block_vector(
            num_insts=length,
            ratios=ratios,
            dependency_flags=dep_flags,
            depth=depth
        )


# 使用示例
if __name__ == "__main__":
    # 初始化Fuzzer
    fuzzer = EnhancedFuzzer(
        instruction_avg_loss={
            'add': 0.12, 'addi': 0.14, 'sub': 0.11,
            'beq': 0.08, 'bne': 0.09,
            'mul': 0.15, 'div': 0.17,
            'ld': 0.22, 'lw': 0.20,
            'sd': 0.18, 'sw': 0.16
        },
        instruction_counts={
            'add': 1200, 'addi': 1500, 'sub': 800,
            'beq': 600, 'bne': 500,
            'mul': 400, 'div': 300,
            'ld': 900, 'lw': 700,
            'sd': 600, 'sw': 500
        },
        block_length_avg_loss={
            "3": 0.05, "5": 0.12, "8": 0.18, "10": 0.14, "15": 0.08
        },
        block_length_counts={
            "3": 817, "5": 554, "8": 342, "10": 167, "15": 85
        }
    )

    # 生成测试用例
    for i in range(100):
        test_block = fuzzer.generate()

        # 每轮迭代
        if i % 10 == 0:
            # 假设这里从模型中获取新的loss信息
            new_instruction_loss = {
                # 模拟的新loss数据
                'add': 0.13, 'addi': 0.12, 'sub': 0.12,
                'beq': 0.09, 'bne': 0.08,
                'mul': 0.16, 'div': 0.15,
                'ld': 0.21, 'lw': 0.19,
                'sd': 0.19, 'sw': 0.17
            }

            new_block_length_loss = {
                "3": 0.06, "5": 0.11, "8": 0.19, "10": 0.15, "15": 0.07, "12": 0.14
            }

            # 更新策略
            fuzzer.update_strategy(
                new_instruction_loss,
                new_block_length_loss,
                avg_loss=0.14
            )

            # 打印当前状态
            print(f"迭代 {i}, 温度: {fuzzer.temp:.2f}, 探索率: {fuzzer.explore_rate:.2f}")


    # def test2(len_bb):
#     block = gen_block(len_bb)
#     print(block)
#     analyzer = DependencyAnalyzer()
#     raw, war, waw = analyzer.analyze(block)
#     print(f"Analysis results: RAW={raw}, WAR={war}, WAW={waw}")
#     # analyzer.print_summary()
#     # print()
#     # analyzer.print_details()
#     return block


    # file_path = "experiments/transformer_v1_20250312_223609/analysis_epoch_8/analysis_summary.json"
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    #
    # os.makedirs('./random_generate/asm', exist_ok=True)
    #
    # num_bb = 2000
    # num = 0
    # for key, value in data["block_dict"].items():
    #     cnt = round(value * num_bb)
    #     # 生成指定长度基本块的个数
    #     for i in range(cnt):
    #         block = gen_block_vector(normalized_vector = data["instruction_vec"], len_bb = int(key))
    #
    #         with open(f'./random_generate/asm/test{num}_nojump.S', 'w') as file:
    #             # file.write("# LLVM-MCA-BEGIN A simple example" + '\n')
    #             for line in block:
    #                 file.write(line.code + '\n')
    #             # file.write("# LLVM-MCA-END")
    #         num += 1

    fuzzer = HybridFuzzer(
        type_loss={
            'shifts_arithmetic': 0.12,
            'compare': 0.08,
            'mul_div': 0.15,
            'load': 0.22,
            'store': 0.18
        },
        length_loss=block_length_avg_loss,
        length_counts=block_length_counts
    )

    # 每轮迭代
    new_type_loss = ...  # 从模型获取
    new_length_data = ...
    fuzzer.update_strategy(new_type_loss, new_length_data)

    # 生成测试用例
    for _ in range(100):
        test_block = fuzzer.generate()
