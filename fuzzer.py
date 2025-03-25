import pprint as pp
import sys
import json
from rvmca.gen import gen_block
from rvmca.gen.inst_gen import gen_block_vector, DependencyAnalyzer
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
                 temp: float = 0.8,
                 explore_rate: float = 0.1,
                 long_block_penalty: float = 0.1,
                 long_block_threshold: int = 128):
        """
        初始化Fuzzer

        Args:
            instruction_avg_loss: 各指令类型的平均loss字典
            instruction_counts: 各指令类型的出现次数字典
            block_length_avg_loss: 各基本块长度的平均loss字典
            block_length_counts: 各基本块长度的出现次数字典
            temp: 温度系数，控制loss影响的强度
            explore_rate: 探索率，控制随机探索的比例
            long_block_penalty: 长基本块惩罚系数(1.0表示无惩罚，0.1表示降低到10%)
            long_block_threshold: 长基本块阈值，超过此长度的基本块将被惩罚
        """
        # 控制参数 - 必须首先初始化，因为后续方法会使用这些参数
        self.temp = temp  # 温度系数
        self.explore_rate = explore_rate  # 探索率
        self.depen_boost = [0.5, 0.5, 0.5]  # 依赖关系增强因子 [WAW, RAW, WAR]
        self.long_block_penalty = long_block_penalty  # 长基本块惩罚系数
        self.long_block_threshold = long_block_threshold  # 长基本块阈值

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
        将指令级别的loss聚合到类别级别，同时考虑原始分布

        Args:
            instr_loss: 指令级别的loss字典
            instr_counts: 指令级别的计数字典

        Returns:
            类别级别的loss权重数组
        """
        # 初始化类别loss累加器和计数累加器
        category_loss_sum = {cat: 0.0 for cat in self.type_order}
        category_count_sum = {cat: 0 for cat in self.type_order}

        # 按类别聚合loss和计数
        for instr, loss in instr_loss.items():
            category = self.instr_categories.get(instr)
            if category and category in self.type_order:
                count = instr_counts.get(instr, 1)
                category_loss_sum[category] += loss
                category_count_sum[category] += count

        # 计算每个类别的平均loss和原始分布比例
        category_avg_loss = []
        category_orig_ratio = []
        total_count = sum(category_count_sum.values())

        for cat in self.type_order:
            # 计算平均loss
            if category_count_sum[cat] > 0:
                avg_loss = category_loss_sum[cat] / category_count_sum[cat]
            else:
                avg_loss = 0.01  # 对于没有数据的类别，使用默认值
            category_avg_loss.append(avg_loss)

            # 计算原始分布比例
            if total_count > 0:
                orig_ratio = category_count_sum[cat] / total_count
            else:
                orig_ratio = 0.2  # 默认均匀分布
            category_orig_ratio.append(orig_ratio)

        # 转换为numpy数组
        avg_loss_array = np.array(category_avg_loss)
        # print(avg_loss_array/avg_loss_array.sum())
        orig_ratio_array = np.array(category_orig_ratio)
        print(category_orig_ratio/orig_ratio_array.sum())

        # 将平均loss转换为权重（归一化）
        if avg_loss_array.sum() > 0:
            loss_weights = avg_loss_array / avg_loss_array.sum()
        else:
            loss_weights = np.ones_like(avg_loss_array) / len(avg_loss_array)

        # 确保原始比例归一化
        if orig_ratio_array.sum() > 0:
            orig_ratio_array = orig_ratio_array / orig_ratio_array.sum()
        else:
            orig_ratio_array = np.ones_like(orig_ratio_array) / len(orig_ratio_array)

        # 平衡loss和原始分布 (beta控制两者的平衡)
        beta = 0.99  # 0.5表示平均考虑loss和原始分布
        # print("loss_weights",loss_weights)
        # print("orig_ratio_array",orig_ratio_array)
        combined_weights = (1 - beta) * loss_weights + beta * orig_ratio_array
        # print(combined_weights)

        # 应用最小阈值，确保每种类型都有一定表示
        min_threshold = 0.005
        combined_weights = np.maximum(combined_weights, min_threshold)

        # 重新归一化
        return combined_weights / combined_weights.sum()

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

        # 确保键类型一致，全部转为字符串
        length_loss_str = {str(k): v for k, v in length_loss.items()}
        length_counts_str = {str(k): v for k, v in length_counts.items()}

        total_count = sum(float(c) for c in length_counts_str.values())

        # 合并两个字典的键集合
        all_lengths = set(length_loss_str.keys()) | set(length_counts_str.keys())

        for l_str in all_lengths:
            try:
                length = int(l_str)
                count = length_counts_str.get(l_str, 0)
                loss = length_loss_str.get(l_str, 0.1)  # 默认loss值

                # 确保count是数值类型
                if not isinstance(count, (int, float)):
                    print(f"警告: 长度 {l_str} 的计数不是数值: {count}")
                    count = 0

                # 根据样本数量计算置信度
                confidence = min(1.0, float(count) / 100)  # 样本数超过100时置信度为1.0
                # 计算平均loss，避免除以零
                orig_dist_weight = count / total_count if total_count > 0 else 0
                avg_loss = loss / max(1, count) if count > 0 else 0.01
                alpha = 0.6  # 增大这个值会更强调原始分布
                weighted_loss = (1 - alpha) * avg_loss * (0.3 + 0.7 * confidence) + alpha * orig_dist_weight
                # weighted_loss = avg_loss * (0.3 + 0.7 * confidence)
                # 存储长度统计信息
                length_stats[length] = {
                    'loss': loss,
                    'count': count,
                    'confidence': confidence,
                    'weighted_loss': weighted_loss  # 加权loss
                }
            except (ValueError, TypeError) as e:
                print(f"处理长度 {l_str} 时出错: {e}")

        return length_stats

    def _build_length_distribution(self, length_stats: Dict[int, Dict]) -> Dict[int, float]:
        """
        构建长度概率分布，对长基本块应用惩罚

        Args:
            length_stats: 长度统计信息

        Returns:
            长度到选择概率的映射
        """
        if not length_stats:
            # 默认均匀分布
            default_lengths = [4, 8, 12, 16, 20]
            return {l: 1.0 / len(default_lengths) for l in default_lengths}

        # 提取加权loss并计算初始softmax分布
        lengths = list(length_stats.keys())
        weighted_losses = [stats['weighted_loss'] for stats in length_stats.values()]
        # for length, count in sorted(length_stats.items()):
        #     print(length,": ", count['weighted_loss'])

        # 应用温度缩放的softmax
        exp_losses = np.exp(np.array(weighted_losses) / self.temp)
        initial_probs = exp_losses / exp_losses.sum()

        # 对长基本块应用惩罚
        final_probs = initial_probs.copy()
        for i, length in enumerate(lengths):
            if length > self.long_block_threshold * 0.5:
                penalty_factor = 1.0 - (1.0 - self.long_block_penalty) * min(1.0, (
                            length - self.long_block_threshold * 0.5) / (self.long_block_threshold * 0.5))

                final_probs[i] *= penalty_factor

        # 重新归一化概率分布
        if final_probs.sum() > 0:
            final_probs = final_probs / final_probs.sum()
        else:
            # 如果所有概率都被惩罚到接近0，则使用均匀分布
            final_probs = np.ones_like(final_probs) / len(final_probs)

        return {length: prob for length, prob in zip(lengths, final_probs)}

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

        # 兼顾loss和频率的双重排序
        frequency_sorted = sorted(
            [(length, stats['count']) for length, stats in length_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )

        loss_sorted = sorted(
            [(length, stats['weighted_loss']) for length, stats in length_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # 选择top_n个频率最高的长度，和top_n个loss最高的长度
        top_n = min(2, len(loss_sorted))
        clusters = []
        for i in range(min(2, len(frequency_sorted))):
            center = frequency_sorted[i][0]
            # 确定扩散范围（基于附近长度的可用性）
            nearby_lengths = [l for l in length_stats.keys() if abs(l - center) <= 5]
            if len(nearby_lengths) > 1:
                spread = max(2.0, np.std(nearby_lengths))
            else:
                spread = 3.0  # 默认扩散范围

            clusters.append((center, spread))

        # 添加高loss长度聚类
        for i in range(min(2, len(loss_sorted))):
            center = loss_sorted[i][0]
            if center not in [c[0] for c in clusters]:  # 避免重复
                # 确定扩散范围
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
        # print("base_ratios", base_ratios)

        # 如果处于探索模式，添加随机噪声
        if random.random() < self.explore_rate:
            noise = np.random.uniform(-0.2, 0.2, len(base_ratios))
            # 确保不会产生负值或过高的值
            ratios = np.clip(base_ratios + noise, 0.005, 0.7)
            # 重新归一化
            ratios = ratios / ratios.sum()
        else:
            ratios = base_ratios
        # print("ratios", ratios)

        return ratios.tolist()

    def _choose_length(self) -> int:
        """
        智能长度选择，限制长基本块生成概率

        Returns:
            选择的长度值
        """
        # 探索与利用的策略选择
        if random.random() < self.explore_rate:
            # 探索模式：尝试更广泛的长度范围，但限制生成长基本块
            if random.random() < 0.7 and self.length_clusters:
                # 从随机选择的聚类中采样，但更大的变异
                cluster_idx = random.randint(0, len(self.length_clusters) - 1)
                center, spread = self.length_clusters[cluster_idx]

                # 如果中心点已经超过阈值，使用一个较小的中心点
                if center > self.long_block_threshold * 0.8:
                    center = self.long_block_threshold // 8

                length = max(1, int(np.random.normal(center, spread * 1.5)))

                # 对超长基本块应用额外截断
                if length > self.long_block_threshold:
                    # 根据惩罚系数，有可能将长度截断到阈值以下
                    if random.random() > self.long_block_penalty:
                        length = self.long_block_threshold - random.randint(0, self.long_block_threshold // 2)
            else:
                # 增加对短基本块的偏好
                if random.random() < 0.99:  # 50%概率生成短基本块
                    # 生成更多短基本块 (长度3-10)
                    length = random.randint(3, 10)
                else:
                    # 有时尝试完全新的长度（使用截断的长尾分布）
                    length = max(1, min(self.long_block_threshold * 1.5,
                                        int(np.random.lognormal(1.5, 0.6))))

                    # 对超长基本块应用额外截断
                    if length > self.long_block_threshold:
                        # 根据惩罚系数，有可能将长度截断到阈值以下
                        if random.random() > self.long_block_penalty:
                            length = self.long_block_threshold - random.randint(0, self.long_block_threshold // 2)

        else:
            # 利用模式：使用学习到的分布
            if self.length_probs and random.random() < 0.99:
                # 直接从分布中采样
                lengths = list(self.length_probs.keys())
                probs = list(self.length_probs.values())
                length = np.random.choice(lengths, p=probs)
            else:
                # 从聚类中采样
                if self.length_clusters:
                    # 从随机选择的聚类中采样
                    cluster_idx = random.randint(0, len(self.length_clusters) - 1)
                    center, spread = self.length_clusters[cluster_idx]

                    # 如果中心点已经超过阈值，使用一个较小的中心点
                    if center > self.long_block_threshold * 0.8:
                        center = self.long_block_threshold // 8

                    length = max(1, int(np.random.normal(center, spread)))
                else:
                    # 使用默认值
                    length = random.choice([4, 8, 12, 16])

        # 增加对短基本块的倾向性
        if random.random() < 0.99:  # 40%的概率应用短基本块偏好
            length_bias = max(3, int(length * 0.3))  # 将长度缩小到原来的30%
            length = min(length, length_bias)

        # 对最终的长度做一次额外的截断检查
        if length > self.long_block_threshold:
        # 最终生成长基本块的概率受惩罚系数控制
            if random.random() > self.long_block_penalty * 0.4:  # 额外降低长基本块概率
                length = self.long_block_threshold - random.randint(0, self.long_block_threshold // 2)

        return length

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

        # 依赖概率计算 - 基于指令类型特性 self.depen_boost = [0.5, 0.5, 0.5]
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
        计算适当的深度参数，确保深度不会超出基本块长度

        Args:
            length: 基本块长度

        Returns:
            深度参数
        """
        # 增强的深度计算逻辑
        if length <= 3:
            return min(1, length - 1) if length > 1 else 0
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
                        instruction_counts: Dict[str, int],
                        block_length_counts: Dict[str, int],
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
            instruction_avg_loss, instruction_counts)

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

        # 使用self.length_stats作为长度计数字典
        length_counts_dict = {str(k): stats['count'] for k, stats in self.length_stats.items()}

        # 更新长度统计和分布
        new_length_stats = self._process_length_stats(
            block_length_avg_loss, block_length_counts)

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

    def plan_generation(self, num_blocks: int) -> Dict[int, int]:
        """
        规划生成指定数量的基本块时，各长度的分布情况

        Args:
            num_blocks: 要生成的基本块数量

        Returns:
            长度到数量的映射字典
        """
        length_distribution = {}

        # 模拟选择过程，统计各长度出现次数
        for _ in range(num_blocks):
            length = self._choose_length()
            # 确保长度至少为3，以防止依赖关系创建出错
            length = max(3, length)

            # 更新分布字典
            if length in length_distribution:
                length_distribution[length] += 1
            else:
                length_distribution[length] = 1

        # 按长度排序
        return dict(sorted(length_distribution.items()))

    def generate(self, preview_plan: bool = False) -> List:
        """
        生成单个测试基本块

        Args:
            preview_plan: 是否预览生成计划 (用于单个基本块生成没有实际意义)

        Returns:
            生成的基本块向量表示
        """
        # 自适应选择参数
        ratios = self._adapt_type_ratios()
        length = self._choose_length()

        # 确保长度至少为3，以防止依赖关系创建出错
        length = max(3, length)

        dep_flags = self._gen_dependency_flags()
        depth = self._calculate_depth(length)

        # 确保深度不超过长度-1，以防止索引越界
        depth = min(depth, length - 1)

        # 生成基本块向量
        return gen_block_vector(
            num_insts=length,
            ratios=ratios,
            dependency_flags=dep_flags,
            depth=depth
        )

    def generate_multiple(self, num_blocks: int) -> Tuple[Dict[int, int], List[List]]:
        """
        生成多个测试基本块，并返回长度分布信息

        Args:
            num_blocks: 要生成的基本块数量

        Returns:
            (长度分布字典, 生成的基本块列表)
        """
        # 先规划长度分布
        length_distribution = self.plan_generation(num_blocks)

        # 按照规划生成基本块
        blocks = []
        for length, count in length_distribution.items():
            for _ in range(count):
                # 自适应选择参数
                ratios = self._adapt_type_ratios()
                dep_flags = self._gen_dependency_flags()
                depth = self._calculate_depth(length)

                # 确保深度不超过长度-1，以防止索引越界
                depth = min(depth, length - 1)

                # 生成基本块并添加到列表
                block = gen_block_vector(
                    num_insts=length,
                    ratios=ratios,
                    dependency_flags=dep_flags,
                    depth=depth
                )
                blocks.append(block)

        # 打乱基本块顺序，避免连续的相同长度基本块
        random.shuffle(blocks)

        return length_distribution, blocks

import os
import shutil

def rm_all_files(directory: str):
    if os.path.exists(directory):
        # 遍历目录中的所有文件
        for filename in os.listdir(directory):
            # 构建文件的完整路径
            file_path = os.path.join(directory, filename)
            try:
                # 如果是文件，则删除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # 如果是目录，则递归删除（如果需要）
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Directory {directory} does not exist.")

if __name__ == "__main__":
    rm_all_files("./random_generate/asm/")
    rm_all_files("./random_generate/binary/")

    import argparse
    parser = argparse.ArgumentParser(description="basic block generator")
    parser.add_argument("-n", type=int, default=100, help="number of basic blocks to generate")
    args = parser.parse_args()

    with open('experiments/lstm_exp2_20250325_091432/statistics/loss_stats_epoch_45.json', 'r', encoding='utf-8') as f:
    # with open('experiments/incremental_transformer_20250324_173953/statistics/loss_stats_epoch_5.json','r', encoding='utf-8') as f:
        data = json.load(f)

    fuzzer = EnhancedFuzzer(instruction_avg_loss=data['type_avg_loss'],
                            instruction_counts=data['type_counts'],
                            block_length_avg_loss=data['block_length_avg_loss'],
                            block_length_counts=data['block_length_counts'],
                            long_block_penalty=0.1,  # 强烈惩罚长基本块，将概率降至原来的10%
                            temp=0.25,
                            long_block_threshold=128)



    print(f"规划生成 {args.n} 个基本块...")

    # 预览长度分布计划
    length_plan = fuzzer.plan_generation(args.n)
    print("基本块长度分布计划:")
    cnt = 0
    for length, count in length_plan.items():
        if int(length) <21:
            cnt += count
        print(f"  长度 {length}: {count} 个")
    print(cnt)
    # 按照计划生成基本块
    print("\n开始生成基本块...")
    blocks = []
    for length, count in length_plan.items():
        for i in range(count):
            block = fuzzer.generate()  # 生成单个基本块
            with open(f'./random_generate/asm/test{len(blocks)}_{len(block)}_nojump.S', 'w') as file:
            # file.write("# LLVM-MCA-BEGIN A simple example" + '\n')
                for line in block:
                    file.write(line.code + '\n')
                # file.write("# LLVM-MCA-END")
            blocks.append(block)

    print(f"成功生成 {len(blocks)} 个基本块")

    # # experiments/incremental_transformer_20250324_173953/statistics/loss_stats_epoch_5.json
    # with open('experiments/incremental_transformer_20250324_173953/statistics/loss_stats_epoch_5.json', 'r', encoding='utf-8') as f:
    #     new_data = json.load(f)
    #
    # fuzzer.update_strategy(
    #     instruction_avg_loss=new_data['type_avg_loss'],
    #     instruction_counts=new_data['type_counts'],
    #     block_length_avg_loss=new_data['block_length_avg_loss'],
    #     block_length_counts=new_data['block_length_counts'],
    #     avg_loss=2.614285
    # )
    #
    # print("\n策略更新后的基本块长度分布计划:")
    # new_length_plan = fuzzer.plan_generation(args.n)
    # for length, count in new_length_plan.items():
    #     print(f"  长度 {length}: {count} 个")
    #
    # # 比较前后变化
    # print("\n分布变化分析:")
    # all_lengths = set(list(length_plan.keys()) + list(new_length_plan.keys()))
    # for length in sorted(all_lengths):
    #     before = length_plan.get(length, 0)
    #     after = new_length_plan.get(length, 0)
    #     change = after - before
    #     change_symbol = "+" if change > 0 else ""
    #     print(f"  长度 {length}: {before} → {after} ({change_symbol}{change})")
    #
    # print("\n开始生成基本块...")
    # blocks = []
    # for length, count in new_length_plan.items():
    #     for i in range(count):
    #         block = fuzzer.generate()  # 生成单个基本块
    #         with open(f'./random_generate/asm/test{len(blocks)}_{len(block)}_nojump.S', 'w') as file:
    #         # file.write("# LLVM-MCA-BEGIN A simple example" + '\n')
    #             for line in block:
    #                 file.write(line.code + '\n')
    #             # file.write("# LLVM-MCA-END")
    #         blocks.append(block)
    #
    # print(f"成功生成 {len(blocks)} 个基本块")

#     # def test2(len_bb):
#     block = gen_block(len_bb)
#     print(block)
#     analyzer = DependencyAnalyzer()
#     raw, war, waw = analyzer.analyze(block)
#     print(f"Analysis results: RAW={raw}, WAR={war}, WAW={waw}")
#     # analyzer.print_summary()
#     # print()
#     # analyzer.print_details()
#     return block
