import pprint as pp
import sys
import json
from rvmca.gen import gen_block
from rvmca.gen.inst_gen import gen_block_vector, DependencyAnalyzer
import numpy as np
import random
from typing import Dict, List, Tuple
import os
import shutil
import argparse

class EnhancedFuzzer:
    """
    Enhanced Intelligent Fuzzer that uses loss feedback to generate targeted test basic blocks

    Learns from loss distributions across instruction types and block lengths to adjust generation strategies
    while maintaining randomness to explore broader test spaces
    """

    def __init__(self,
                 instruction_avg_loss: Dict[str, float],
                 instruction_counts: Dict[str, int],
                 block_length_avg_loss: Dict[str, float],
                 block_length_counts: Dict[str, int],
                 temp: float = 0.8,
                 explore_rate: float = 0.05,
                 long_block_penalty: float = 0.1,
                 long_block_threshold: int = 64):
        """
        Initialize the Fuzzer

        Args:
            instruction_avg_loss: Dictionary of average loss by instruction type
            instruction_counts: Dictionary of occurrence counts by instruction type
            block_length_avg_loss: Dictionary of average loss by block length
            block_length_counts: Dictionary of occurrence counts by block length
            temp: Temperature coefficient controlling the strength of loss influence
            explore_rate: Exploration rate controlling the proportion of random exploration
            long_block_penalty: Long block penalty coefficient (1.0 means no penalty, 0.1 means reduced to 10%)
            long_block_threshold: Long block threshold beyond which blocks will be penalized
        """
        # Control parameters - initialize first since later methods use these parameters
        self.temp = temp  # Temperature coefficient
        self.explore_rate = explore_rate  # Exploration rate
        self.depen_boost = [0.5, 0.5, 0.5]  # Dependency relationship boost factor [WAW, RAW, WAR]
        self.long_block_penalty = long_block_penalty  # Long block penalty coefficient
        self.long_block_threshold = long_block_threshold  # Long block threshold

        # Store original length distribution for stronger adherence
        self.original_length_distribution = self._normalize_length_distribution(block_length_counts)

        # Instruction category mapping
        # self.instr_categories = {
        #     # Shifts and arithmetic operations
        #     'add': 'shifts_arithmetic', 'addi': 'shifts_arithmetic',
        #     'sub': 'shifts_arithmetic', 'sll': 'shifts_arithmetic',
        #     'slli': 'shifts_arithmetic', 'srl': 'shifts_arithmetic',
        #     'srli': 'shifts_arithmetic', 'sra': 'shifts_arithmetic',
        #     'srai': 'shifts_arithmetic', 'or': 'shifts_arithmetic',
        #     'ori': 'shifts_arithmetic', 'and': 'shifts_arithmetic',
        #     'andi': 'shifts_arithmetic', 'xor': 'shifts_arithmetic',
        #     'xori': 'shifts_arithmetic',
        #
        #     # Comparison instructions
        #     'slt': 'compare', 'slti': 'compare', 'sltu': 'compare',
        #     'sltiu': 'compare', 'beq': 'compare', 'bne': 'compare',
        #     'blt': 'compare', 'bge': 'compare', 'bltu': 'compare',
        #     'bgeu': 'compare',
        #
        #     # Multiplication and division
        #     'mul': 'mul_div', 'mulh': 'mul_div', 'mulhu': 'mul_div',
        #     'mulhsu': 'mul_div', 'div': 'mul_div', 'divu': 'mul_div',
        #     'rem': 'mul_div', 'remu': 'mul_div',
        #
        #     # Load instructions
        #     'lb': 'load', 'lh': 'load', 'lw': 'load', 'ld': 'load',
        #     'lbu': 'load', 'lhu': 'load', 'lwu': 'load',
        #
        #     # Store instructions
        #     'sb': 'store', 'sh': 'store', 'sw': 'store', 'sd': 'store'
        # }
        self.instr_categories = {
            # Arithmetic
            'add': 'arithmetic', 'addi': 'arithmetic',
            'addw': 'arithmetic', 'addiw': 'arithmetic',
            'sub': 'arithmetic', 'subw': 'arithmetic',
            'lui': 'arithmetic', 'auipc': 'arithmetic',  # 8

            # Shifts
            'sll': 'shifts', 'sllw': 'shifts',
            'slli': 'shifts', 'slliw': 'shifts',
            'srl': 'shifts', 'srlw': 'shifts',
            'srli': 'shifts', 'srliw': 'shifts',
            'sra': 'shifts', 'sraw': 'shifts',
            'srai': 'shifts', 'sraiw': 'shifts',  # 12

            # Logical
            'or': 'logical', 'ori': 'logical',
            'xor': 'logical', 'xori': 'logical',
            'and': 'logical', 'andi': 'logical',  # 6

            # Comparison instructions
            'slt': 'compare', 'slti': 'compare', 'sltu': 'compare', 'sltiu': 'compare',  # 4

            # Multiplication
            'mul': 'mul', 'mulh': 'mul', 'mulhu': 'mul', 'mulhsu': 'mul', 'mulw': 'mul',  # 5

            # Division
            'div': 'div', 'divu': 'div', 'divw': 'div', 'divuw': 'div',  # 4

            # Remainder
            'rem': 'rem', 'remu': 'rem', 'remw': 'rem', 'remuw': 'rem',  # 4

            # Load instructions
            'lb': 'load', 'lh': 'load', 'lw': 'load', 'ld': 'load',
            'lbu': 'load', 'lhu': 'load', 'lwu': 'load',  # 7

            # Store instructions
            'sb': 'store', 'sh': 'store', 'sw': 'store', 'sd': 'store'  # 4
        }


        # Instruction type configuration
        self.type_order = ['arithmetic', 'shifts', 'logical', 'compare', 'mul', 'div', 'rem', 'load', 'store']
        self.type_weights = self._aggregate_instruction_loss(instruction_avg_loss, instruction_counts)

        # Length strategy configuration
        self.length_stats = self._process_length_stats(block_length_avg_loss, block_length_counts)
        self.length_probs = self._build_length_distribution(self.length_stats)
        self.length_clusters = self._identify_length_clusters(self.length_stats)

        # Performance tracking
        self.perf_history = []  # Track average loss history
        self.update_counter = 0  # Update counter

        # Save original data
        self.instruction_loss_map = instruction_avg_loss.copy()
        self.instruction_count_map = instruction_counts.copy()

    def _normalize_length_distribution(self, length_counts: Dict[str, int]) -> Dict[int, float]:
        """
        Convert raw length counts to a normalized probability distribution

        Args:
            length_counts: Dictionary with length (as string) to count mapping

        Returns:
            Dictionary with length (as int) to probability mapping
        """
        # Convert keys to integers when possible
        int_counts = {}
        for k, v in length_counts.items():
            try:
                int_counts[int(k)] = v
            except (ValueError, TypeError):
                print(f"Warning: Could not convert length key '{k}' to integer")
                continue

        # Calculate total and normalize
        total = sum(int_counts.values())
        if total == 0:
            # Default uniform distribution if no counts
            lens = list(range(4, 25, 4))
            return {l: 1.0 / len(lens) for l in lens}

        # Return normalized distribution
        return {k: v / total for k, v in int_counts.items()}

    def _aggregate_instruction_loss(self,
                                    instr_loss: Dict[str, float],
                                    instr_counts: Dict[str, int]) -> np.ndarray:
        """
        Aggregate instruction-level loss to category level, considering original distribution

        Args:
            instr_loss: Dictionary of instruction-level loss
            instr_counts: Dictionary of instruction-level counts

        Returns:
            Category-level loss weight array
        """
        # Initialize category loss and count accumulators
        category_loss_sum = {cat: 0.0 for cat in self.type_order}
        category_count_sum = {cat: 0 for cat in self.type_order}

        # Aggregate loss and counts by category
        for instr, loss in instr_loss.items():
            category = self.instr_categories.get(instr)
            if category and category in self.type_order:
                count = instr_counts.get(instr, 1)
                category_loss_sum[category] += loss
                category_count_sum[category] += count

        # Calculate average loss and original distribution ratio for each category
        category_avg_loss = []
        category_orig_ratio = []
        total_count = sum(category_count_sum.values())

        for cat in self.type_order:
            # Calculate average loss
            if category_count_sum[cat] > 0:
                avg_loss = category_loss_sum[cat] / category_count_sum[cat]
            else:
                avg_loss = 0.01  # Default value for categories with no data
            category_avg_loss.append(avg_loss)

            # Calculate original distribution ratio
            if total_count > 0:
                orig_ratio = category_count_sum[cat] / total_count
            else:
                orig_ratio = 0.2  # Default uniform distribution
            category_orig_ratio.append(orig_ratio)

        # Convert to numpy arrays
        avg_loss_array = np.array(category_avg_loss)
        print(avg_loss_array)
        orig_ratio_array = np.array(category_orig_ratio)

        # Convert average loss to weights (normalize)
        if avg_loss_array.sum() > 0:
            loss_weights = avg_loss_array / avg_loss_array.sum()
        else:
            loss_weights = np.ones_like(avg_loss_array) / len(avg_loss_array)

        # Ensure original ratios are normalized
        if orig_ratio_array.sum() > 0:
            orig_ratio_array = orig_ratio_array / orig_ratio_array.sum()
        else:
            orig_ratio_array = np.ones_like(orig_ratio_array) / len(orig_ratio_array)

        print("loss_weights", loss_weights)
        print("orig_ratio_array", orig_ratio_array)

        # Balance loss and original distribution (beta controls the balance)
        beta = 0.95  # 0.5 means equal consideration of loss and original distribution
        # print("loss_weights",loss_weights)
        # print("orig_ratio_array",orig_ratio_array)
        combined_weights = (1 - beta) * loss_weights + beta * orig_ratio_array
        # print(combined_weights)

        # Apply minimum threshold to ensure every type has representation
        min_threshold = 0.005
        combined_weights = np.maximum(combined_weights, min_threshold)

        # Renormalize
        return combined_weights / combined_weights.sum()

    def _process_length_stats(self,
                              length_loss: Dict[str, float],
                              length_counts: Dict[str, int]) -> Dict[int, Dict]:
        """
        Process length statistics into a more usable format

        Args:
            length_loss: Dictionary mapping length to loss
            length_counts: Dictionary mapping length to count

        Returns:
            Dictionary mapping length to statistics dictionary
        """
        length_stats = {}

        # Ensure key types are consistent, convert all to strings
        length_loss_str = {str(k): v for k, v in length_loss.items()}
        length_counts_str = {str(k): v for k, v in length_counts.items()}

        total_count = sum(float(c) for c in length_counts_str.values())

        # Merge key sets from both dictionaries
        all_lengths = set(length_loss_str.keys()) | set(length_counts_str.keys())

        for l_str in all_lengths:
            try:
                length = int(l_str)
                count = length_counts_str.get(l_str, 0)
                loss = length_loss_str.get(l_str, 0.1)  # Default loss value

                # Ensure count is a numeric type
                if not isinstance(count, (int, float)):
                    print(f"Warning: Count for length {l_str} is not numeric: {count}")
                    count = 0

                # Calculate confidence based on sample size
                confidence = min(1.0, float(count) / 100)  # Confidence is 1.0 when sample count exceeds 100

                # Calculate original distribution weight (much higher emphasis now)
                orig_dist_weight = count / total_count if total_count > 0 else 0

                # Calculate average loss, avoiding division by zero
                avg_loss = loss / max(1, count) if count > 0 else 0.01

                # Balance between original distribution and loss (increased alpha for stronger original distribution influence)
                alpha = 0.8  # Increased from 0.6 to 0.8 - higher value emphasizes original distribution more
                weighted_loss = (1 - alpha) * avg_loss * (0.3 + 0.7 * confidence) + alpha * orig_dist_weight

                # Store length statistics
                length_stats[length] = {
                    'loss': loss,
                    'count': count,
                    'confidence': confidence,
                    'weighted_loss': weighted_loss,  # Weighted loss
                    'orig_prob': orig_dist_weight  # Store original probability for direct use
                }
            except (ValueError, TypeError) as e:
                print(f"Error processing length {l_str}: {e}")

        return length_stats

    def _build_length_distribution(self, length_stats: Dict[int, Dict]) -> Dict[int, float]:
        """
        Build length probability distribution, applying penalty to long basic blocks

        Args:
            length_stats: Length statistics dictionary

        Returns:
            Dictionary mapping length to selection probability
        """
        if not length_stats:
            # Default uniform distribution
            default_lengths = [4, 8, 12, 16, 20]
            return {l: 1.0 / len(default_lengths) for l in default_lengths}

        # Extract weighted loss and original probability
        lengths = list(length_stats.keys())
        weighted_losses = [stats['weighted_loss'] for stats in length_stats.values()]
        original_probs = [stats['orig_prob'] for stats in length_stats.values()]

        # Convert to numpy arrays
        weighted_losses = np.array(weighted_losses)
        original_probs = np.array(original_probs)

        # Normalize original probabilities if needed
        if np.sum(original_probs) > 0:
            original_probs = original_probs / np.sum(original_probs)
        else:
            original_probs = np.ones_like(original_probs) / len(original_probs)

        # Apply temperature-scaled softmax to weighted losses
        exp_losses = np.exp(weighted_losses / self.temp)
        loss_based_probs = exp_losses / exp_losses.sum() if exp_losses.sum() > 0 else np.ones_like(exp_losses) / len(
            exp_losses)

        # Combine loss-based probabilities with original distribution (higher weight to original distribution)
        distribution_weight = 0.7  # Control parameter - higher value gives more weight to original distribution
        combined_probs = (1 - distribution_weight) * loss_based_probs + distribution_weight * original_probs

        # Apply long block penalty
        final_probs = combined_probs.copy()
        for i, length in enumerate(lengths):
            if length > self.long_block_threshold * 0.5:
                penalty_factor = 1.0 - (1.0 - self.long_block_penalty) * min(1.0, (
                        length - self.long_block_threshold * 0.5) / (self.long_block_threshold * 0.5))
                final_probs[i] *= penalty_factor

        # Renormalize probability distribution
        if final_probs.sum() > 0:
            final_probs = final_probs / final_probs.sum()
        else:
            # Use uniform distribution if all probabilities penalized to near zero
            final_probs = np.ones_like(final_probs) / len(final_probs)

        return {length: prob for length, prob in zip(lengths, final_probs)}

    def _identify_length_clusters(self, length_stats: Dict[int, Dict]) -> List[Tuple[int, float]]:
        """
        Identify promising length clusters

        Args:
            length_stats: Length statistics dictionary

        Returns:
            List of length clusters, each element is (center point, spread range)
        """
        if not length_stats:
            return [(10, 3.0)]  # Default cluster

        # Sort by both loss and frequency
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

        # Select top_n highest frequency lengths and top_n highest loss lengths
        top_n = min(3, len(loss_sorted))  # Increased from 2 to 3 to better represent the distribution
        clusters = []

        # First add highest frequency lengths
        for i in range(min(3, len(frequency_sorted))):  # Increased from 2 to 3
            center = frequency_sorted[i][0]
            # Determine spread range (based on nearby length availability)
            nearby_lengths = [l for l in length_stats.keys() if abs(l - center) <= 5]
            if len(nearby_lengths) > 1:
                spread = max(2.0, np.std(nearby_lengths))
            else:
                spread = 3.0  # Default spread range

            clusters.append((center, spread))

        # Add high loss length clusters
        for i in range(min(2, len(loss_sorted))):
            center = loss_sorted[i][0]
            if center not in [c[0] for c in clusters]:  # Avoid duplicates
                # Determine spread range
                nearby_lengths = [l for l in length_stats.keys() if abs(l - center) <= 5]
                if len(nearby_lengths) > 1:
                    spread = max(2.0, np.std(nearby_lengths))
                else:
                    spread = 3.0  # Default spread range

                clusters.append((center, spread))

        return clusters

    def _adapt_type_ratios(self) -> List[float]:
        """
        Generate instruction type ratios based on current strategy

        Returns:
            List of ratios for each instruction type
        """
        # Apply temperature scaling
        scaled_weights = np.exp(self.type_weights / self.temp)
        base_ratios = scaled_weights / scaled_weights.sum()
        # print("base_ratios", base_ratios)

        # Add random noise in exploration mode
        if random.random() < self.explore_rate:
            noise = np.random.uniform(-0.2, 0.2, len(base_ratios))
            # Ensure no negative or extremely high values
            ratios = np.clip(base_ratios + noise, 0.005, 0.7)
            # Renormalize
            ratios = ratios / ratios.sum()
        else:
            ratios = base_ratios
        # print("ratios", ratios)

        return ratios.tolist()

    def _choose_length(self) -> int:
        """
        Intelligent length selection, limiting long basic block generation probability

        Returns:
            Selected length value
        """
        # Direct sampling from original distribution with high probability
        if random.random() < 0.6:  # 60% chance to sample directly from original distribution
            # Get original distribution keys and values
            orig_lengths = list(self.original_length_distribution.keys())
            orig_probs = list(self.original_length_distribution.values())

            if orig_lengths and sum(orig_probs) > 0:
                # Sample from original distribution
                return np.random.choice(orig_lengths, p=orig_probs)

        # Exploration vs. exploitation strategy choice
        if random.random() < self.explore_rate:
            # Exploration mode: try broader length range, but limit long basic block generation
            if random.random() < 0.7 and self.length_clusters:
                # Sample from randomly selected cluster, but with more variation
                cluster_idx = random.randint(0, len(self.length_clusters) - 1)
                center, spread = self.length_clusters[cluster_idx]

                # If center point already exceeds threshold, use a smaller center point
                if center > self.long_block_threshold * 0.8:
                    center = self.long_block_threshold // 8

                length = max(1, int(np.random.normal(center, spread * 1.25)))

                # Apply extra truncation for super-long basic blocks
                if length > self.long_block_threshold:
                    # Potential to truncate length below threshold based on penalty coefficient
                    if random.random() > self.long_block_penalty:
                        length = self.long_block_threshold - random.randint(0, self.long_block_threshold // 2)
            else:
                # Preference for short basic blocks
                if random.random() < 0.7:  # Reduced from 0.99 to allow more variety
                    # Generate more short basic blocks (length 3-10)
                    length = random.randint(3, 10)
                else:
                    # Sometimes try completely new lengths (using truncated long-tail distribution)
                    length = max(1, min(self.long_block_threshold * 1.25,
                                        int(np.random.lognormal(1.5, 0.6))))

                    # Apply extra truncation for super-long basic blocks
                    if length > self.long_block_threshold:
                        # Potential to truncate length below threshold based on penalty coefficient
                        if random.random() > self.long_block_penalty:
                            length = self.long_block_threshold - random.randint(0, self.long_block_threshold // 2)

        else:
            # Exploitation mode: use learned distribution
            if self.length_probs and random.random() < 0.8:  # Reduced from 0.99 to allow more cluster sampling
                # Sample directly from distribution
                lengths = list(self.length_probs.keys())
                probs = list(self.length_probs.values())
                length = np.random.choice(lengths, p=probs)
            else:
                # Sample from clusters
                if self.length_clusters:
                    # Sample from randomly selected cluster
                    cluster_idx = random.randint(0, len(self.length_clusters) - 1)
                    center, spread = self.length_clusters[cluster_idx]

                    # If center point already exceeds threshold, use a smaller center point
                    if center > self.long_block_threshold * 0.8:
                        center = self.long_block_threshold // 8

                    length = max(1, int(np.random.normal(center, spread)))
                else:
                    # Use default values
                    length = random.choice([4, 8, 12, 16])

        # Add short block bias (with reduced probability to better maintain original distribution)
        if random.random() < 0.3:  # Reduced from 0.99 to 0.3
            length_bias = max(3, int(length * 0.3))  # Reduce length to 30% of original
            length = min(length, length_bias)

        # Final extra truncation check
        if length > self.long_block_threshold:
            # Final long basic block probability controlled by penalty coefficient
            if random.random() > self.long_block_penalty * 0.4:  # Further reduce long block probability
                length = self.long_block_threshold - random.randint(0, self.long_block_threshold // 2)

        return length

    def _gen_dependency_flags(self) -> List[int]:
        """
        Generate data dependency flags

        Returns:
            Dependency flag list [WAW dependency, RAW dependency, WAR dependency]
            WAW (Write After Write): Two writes to the same register
            RAW (Read After Write): Reading a register after writing to it
            WAR (Write After Read): Writing to a register after reading it
        """
        # Get weights for each instruction type
        arith_weight = self.type_weights[0]  # Arithmetic/shift/logic instruction weight
        compare_weight = self.type_weights[1]  # Compare instruction weight
        mul_div_weight = self.type_weights[2]  # Multiplication/division instruction weight
        load_weight = self.type_weights[3]  # Load instruction weight
        store_weight = self.type_weights[4]  # Store instruction weight

        # Dependency probability calculation - based on instruction type characteristics
        # WAW: Writing to the same register, related to instructions with def registers (arithmetic/store)
        waw_dep_prob = self.depen_boost[0] + 0.3 * (arith_weight + store_weight)

        # RAW: Reading a value written earlier, most instructions might read values (load/arithmetic depend more)
        raw_dep_prob = self.depen_boost[1] + 0.3 * (load_weight + arith_weight + mul_div_weight)

        # WAR: Writing to a register read earlier, often used for register reuse
        war_dep_prob = self.depen_boost[2] + 0.2 * (arith_weight + compare_weight)

        # Limit dependency probabilities to reasonable ranges
        probs = [
            np.clip(waw_dep_prob, 0.1, 0.9),
            np.clip(raw_dep_prob, 0.1, 0.9),
            np.clip(war_dep_prob, 0.1, 0.9)
        ]

        # Generate dependency flags
        return [int(random.random() < p) for p in probs]

    def _calculate_depth(self, length: int) -> int:
        """
        Calculate appropriate depth parameter, ensuring depth doesn't exceed basic block length

        Args:
            length: Basic block length

        Returns:
            Depth parameter
        """
        # Enhanced depth calculation logic
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
        Update strategy based on new loss information

        Args:
            instruction_avg_loss: New instruction type loss dictionary
            block_length_avg_loss: New basic block length loss dictionary
            instruction_counts: New instruction count dictionary
            block_length_counts: New block length count dictionary
            avg_loss: Optional overall average loss
        """
        self.update_counter += 1

        # Update original length distribution
        self.original_length_distribution = self._normalize_length_distribution(block_length_counts)

        # Update instruction type weights
        new_weights = self._aggregate_instruction_loss(
            instruction_avg_loss, instruction_counts)

        # Adaptive learning rate
        lr = 0.3  # Base learning rate

        # Track and adjust learning rate if average loss provided
        if avg_loss is not None:
            self.perf_history.append(avg_loss)

            # Adjust learning rate if enough history data available
            if len(self.perf_history) >= 5:
                recent_losses = self.perf_history[-5:]
                trend = recent_losses[-1] / max(np.mean(recent_losses[:-1]), 1e-5) - 1

                # Adjust learning rate based on trend
                if trend > 0.05:  # Performance significantly improved
                    lr = min(0.5, lr * 1.2)
                elif trend < -0.05:  # Performance significantly deteriorated
                    lr = max(0.1, lr * 0.8)

        # Apply learning rate to update type weights
        self.type_weights = (1 - lr) * self.type_weights + lr * new_weights
        self.type_weights /= self.type_weights.sum()  # Renormalize

        # Use self.length_stats as length count dictionary
        length_counts_dict = {str(k): stats['count'] for k, stats in self.length_stats.items()}

        # Update length statistics and distribution
        new_length_stats = self._process_length_stats(
            block_length_avg_loss, block_length_counts)

        # Update statistics for existing lengths
        for length, new_stats in new_length_stats.items():
            if length in self.length_stats:
                old_stats = self.length_stats[length]

                # Adjust learning rate based on loss change magnitude
                change_magnitude = abs(new_stats['loss'] - old_stats['loss']) / max(old_stats['loss'], 1e-5)
                adaptive_lr = min(0.5, lr * (1 + change_magnitude))

                # Update loss and weighted loss
                old_stats['loss'] = (1 - adaptive_lr) * old_stats['loss'] + adaptive_lr * new_stats['loss']
                old_stats['count'] += 1
                old_stats['confidence'] = min(1.0, old_stats['count'] / 100)
                old_stats['weighted_loss'] = old_stats['loss'] * (0.3 + 0.7 * old_stats['confidence'])
                # Update original probability
                old_stats['orig_prob'] = new_stats['orig_prob']
            else:
                # New length, add directly
                self.length_stats[length] = new_stats

        # Update derived models
        self.length_probs = self._build_length_distribution(self.length_stats)
        self.length_clusters = self._identify_length_clusters(self.length_stats)

        # Periodically update meta-parameters
        if self.update_counter % 10 == 0 and len(self.perf_history) >= 10:
            self._update_meta_parameters()

    def _update_meta_parameters(self):
        """Update temperature and exploration rate based on performance history"""
        if len(self.perf_history) < 10:
            return  # Insufficient data

        # Get recent loss history
        history = self.perf_history
        recent = history[-10:]

        # Check performance trend
        is_improving = recent[-1] > np.mean(recent[:5])
        is_stagnating = abs(recent[-1] - np.mean(recent[:5])) / max(np.mean(recent[:5]), 1e-5) < 0.03

        # Adjust temperature
        if is_improving:
            # If performance improving, gradually reduce temperature
            self.temp = max(0.3, self.temp * 0.95)
        elif is_stagnating:
            # If performance stagnating, more aggressively increase temperature
            self.temp = min(1.0, self.temp * 1.15)
        else:
            # If performance deteriorating, slightly increase temperature
            self.temp = min(0.8, self.temp * 1.05)

        # Adjust exploration rate
        if is_improving:
            # If performance improving, gradually reduce exploration rate
            self.explore_rate = max(0.1, self.explore_rate * 0.9)
        elif is_stagnating:
            # If performance stagnating, more aggressively increase exploration rate
            self.explore_rate = min(0.4, self.explore_rate * 1.2)
        else:
            # If performance deteriorating, slightly increase exploration rate
            self.explore_rate = min(0.3, self.explore_rate * 1.1)

    def plan_generation(self, num_blocks: int) -> Dict[int, int]:
        """
        Plan distribution of lengths when generating specified number of basic blocks

        Args:
            num_blocks: Number of basic blocks to generate

        Returns:
            Dictionary mapping length to count
        """
        length_distribution = {}

        # Direct application of original distribution for 70% of blocks
        orig_block_count = int(num_blocks * 0.7)
        remaining_blocks = num_blocks - orig_block_count

        if orig_block_count > 0 and self.original_length_distribution:
            # Sample from original distribution
            orig_lengths = list(self.original_length_distribution.keys())
            orig_probs = list(self.original_length_distribution.values())

            if orig_lengths and sum(orig_probs) > 0:
                sampled_lengths = np.random.choice(
                    orig_lengths,
                    size=orig_block_count,
                    p=orig_probs
                )

                # Update distribution
                for length in sampled_lengths:
                    if length in length_distribution:
                        length_distribution[length] += 1
                    else:
                        length_distribution[length] = 1

        # Simulate selection process for remaining blocks
        for _ in range(remaining_blocks):
            length = self._choose_length()
            # Ensure length is at least 3 to prevent dependency creation errors
            length = max(3, length)

            # Update distribution dictionary
            if length in length_distribution:
                length_distribution[length] += 1
            else:
                length_distribution[length] = 1

        # Sort by length
        return dict(sorted(length_distribution.items()))

    def generate(self, preview_plan: bool = False) -> List:
        """
        Generate a single test basic block

        Args:
            preview_plan: Whether to preview generation plan (not meaningful for single block generation)

        Returns:
            Vector representation of generated basic block
        """
        # Adaptively select parameters
        ratios = self._adapt_type_ratios()
        length = self._choose_length()

        # Ensure length is at least 3 to prevent dependency creation errors
        length = max(3, length)

        dep_flags = self._gen_dependency_flags()
        depth = self._calculate_depth(length)

        # Ensure depth doesn't exceed length-1 to prevent index out of bounds
        depth = min(depth, length - 1)

        # Generate basic block vector
        return gen_block_vector(
            num_insts=length,
            ratios=ratios,
            dependency_flags=dep_flags,
            depth=depth
        )

    def generate_multiple(self, num_blocks: int) -> Tuple[Dict[int, int], List[List]]:
        """
        Generate multiple test basic blocks and return length distribution information

        Args:
            num_blocks: Number of basic blocks to generate

        Returns:
            (Length distribution dictionary, list of generated basic blocks)
        """
        # First plan length distribution
        length_distribution = self.plan_generation(num_blocks)

        # Generate basic blocks according to plan
        blocks = []
        for length, count in length_distribution.items():
            for _ in range(count):
                # Adaptively select parameters
                ratios = self._adapt_type_ratios()
                dep_flags = self._gen_dependency_flags()
                depth = self._calculate_depth(length)

                # Ensure depth doesn't exceed length-1 to prevent index out of bounds
                depth = min(depth, length - 1)

                # Generate and add basic block to list
                block = gen_block_vector(
                    num_insts=length,
                    ratios=ratios,
                    dependency_flags=dep_flags,
                    depth=depth
                )
                blocks.append(block)

        # Shuffle block order to avoid consecutive blocks of same length
        random.shuffle(blocks)

        return length_distribution, blocks

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

def incre_generator(file_path, val_loss, fuzzer):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fuzzer.update_strategy(
            instruction_avg_loss=data['type_avg_loss'],
            instruction_counts=data['type_counts'],
            block_length_avg_loss=data['block_length_avg_loss'],
            block_length_counts=data['block_length_counts'],
            avg_loss=val_loss
        )

    return fuzzer

def generator(file_path):
    with open(file_path) as f:
        data = json.load(f)

    fuzzer = EnhancedFuzzer(instruction_avg_loss=data['type_avg_loss'],
                            instruction_counts=data['type_counts'],
                            block_length_avg_loss=data['block_length_avg_loss'],
                            block_length_counts=data['block_length_counts'],
                            long_block_penalty=0.1,
                            explore_rate=0.05,
                            temp=0.25,
                            long_block_threshold=128)

    return fuzzer


def generate_blocks(num_blocks, block_length_counts):
    """
    Generate a new dictionary of block lengths based on an existing distribution.

    Args:
        num_blocks: Number of blocks to generate
        block_length_counts: Dictionary mapping length to count

    Returns:
        Dictionary mapping length to generated count
    """
    # Convert all keys to integers
    length_counts = {int(k): v for k, v in block_length_counts.items()}

    # Calculate total count
    total_count = sum(length_counts.values())

    # Create probability distribution
    lengths = list(length_counts.keys())
    probs = [length_counts[l] / total_count for l in lengths]

    # Generate new blocks based on this distribution
    generated_blocks = np.random.choice(lengths, size=num_blocks, p=probs)

    # Count occurrences
    generated_counts = {}
    for length in generated_blocks:
        if length in generated_counts:
            generated_counts[length] += 1
        else:
            generated_counts[length] = 1

    return generated_counts

def riscv_asm_to_hex(assembly_code):
    import subprocess
    import tempfile
    import os
    # 创建临时文件保存汇编代码
    with tempfile.NamedTemporaryFile(suffix='.s', delete=False) as asm_file:
        asm_file.write(assembly_code.encode())
        asm_file_name = asm_file.name

    # 创建临时文件名用于目标文件
    obj_file_name = asm_file_name + '.o'

    try:
        # 使用riscv64-unknown-linux-gnu-as汇编器将汇编代码编译为目标文件
        subprocess.run(['riscv64-unknown-linux-gnu-as', '-march=rv64g', asm_file_name, '-o', obj_file_name], check=True, stderr=subprocess.DEVNULL)

        # 使用riscv64-unknown-linux-gnu-objdump查看目标文件的十六进制内容
        result = subprocess.run(['riscv64-unknown-linux-gnu-objdump', '-d', obj_file_name],
                                capture_output=True, text=True, check=True)

        # 提取十六进制代码
        hex_codes = []
        # print(result.stdout.splitlines())
        for line in result.stdout.splitlines():
            if ':' in line:
                parts = line.split('\t')
                if len(parts) > 1:
                    hex_part = parts[1].strip()
                    if hex_part:
                        hex_codes.append(hex_part)

        return " ".join(hex_codes)

    except subprocess.CalledProcessError as e:
        print(f"Error during compilation: {e}")
        return None
    finally:
        # 清理临时文件
        if os.path.exists(asm_file_name):
            os.remove(asm_file_name)
        if os.path.exists(obj_file_name):
            os.remove(obj_file_name)

if __name__ == "__main__":
    # rm_all_files("./random_generate/asm/")
    # rm_all_files("./random_generate/binary/")

    parser = argparse.ArgumentParser(description="basic block generator")
    parser.add_argument("-n", type=int, default=100, help="number of basic blocks to generate")
    # parser.add_argument("-exp", type=str, default=None, help="experiment name")
    # parser.add_argument("-epoch", type=str, default=None, help="epoch")
    # parser.add_argument("-loss", type=float, default=None, help="validation loss")
    args = parser.parse_args()

    fuzzer = generator('experiments/lstm_20250427_210029/statistics/train_loss_stats_epoch_8.json')
    fuzzer = incre_generator('experiments/incremental_lstm_20250513_082153/statistics/train_loss_stats_epoch_22.json', 0.064751, fuzzer)
    fuzzer = incre_generator('experiments/incremental_lstm_20250515_230133/statistics/train_loss_stats_epoch_11.json', 0.062639, fuzzer)

    length_plan = fuzzer.plan_generation(args.n)

    print("\n开始生成基本块...")
    oprand_count = {cat: 0 for cat in fuzzer.type_order}
    blocks = []
    cnt = 0
    for length, count in length_plan.items():
        # print(length, count)
        if int(length) < 21:
            cnt += count
        for i in range(count):
            block = fuzzer.generate(length)
            blocks.append({"asm": "\\n".join([i.code for i in block])}) # for mca
            # assembly_code = "\n".join([i.code for i in block])
            # blocks.append({"asm": assembly_code,
            #                "binary": riscv_asm_to_hex(assembly_code)}) # for k230
            for instr in [i.code for i in block]:
                type = fuzzer.instr_categories.get(instr.split()[0])
                if type and type in fuzzer.type_order:
                    oprand_count[type] += 1

    # print("cnt less than 21", cnt)
    # print(oprand_count)

    # with open(f'./random_generate/asm.json', 'w') as file:
    #     json.dump(blocks, file, indent=2) #for mca
    rm_all_files("./random_generate/")
    total_chunks = 10
    chunk_size = len(blocks) // total_chunks
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(blocks))  # Ensure we don't go beyond the list length

        # Create filename with chunk number
        filename = f"./random_generate/asm{i}.json"

        # Write chunk to file
        with open(filename, 'w') as f:
            json.dump(blocks[start_idx:end_idx], f, indent=2)

        print(f"Saved chunk {i + 1}/{total_chunks}: items {start_idx} to {end_idx}")
    # rm_all_files("./random_generate/starfive/")
    # chunk_size = 1000
    # total_chunks = (len(blocks) + chunk_size - 1) // chunk_size
    # for i in range(total_chunks):
    #     start_idx = i * chunk_size
    #     end_idx = min((i + 1) * chunk_size, len(blocks))  # Ensure we don't go beyond the list length
    #
    #     # Create filename with chunk number
    #     filename = f"./random_generate/starfive/asm{i + 1}.json"
    #
    #     # Write chunk to file
    #     with open(filename, 'w') as f:
    #         json.dump(blocks[start_idx:end_idx], f, indent=2)
    #
    #     print(f"Saved chunk {i + 1}/{total_chunks}: items {start_idx} to {end_idx}")


#-----------------------------------------------------------------------------------------------------------------------------
    # with open('experiments/incremental_lstm_20250413_103441/statistics/train_loss_stats_epoch_4.json') as f:
    #     data = json.load(f)
    # new_blocks = generate_blocks(args.n, data['block_length_counts'])
    #
    # blocks = []
    # cnt = 0
    # for length, count in new_blocks.items():
    #     print(length, count)
    #     if int(length) < 21:
    #         cnt += count
    #     for i in range(count):
    #         block = gen_block(length)
    #         blocks.append({"asm": "\\n".join([i.code for i in block])})
    # with open(f'./random_generate/asm.json', 'w') as file:
    #     json.dump(blocks, file, indent=2)
    # print(cnt)


    # for i in range(1000):
    #     block = gen_block(random.randint(2,15))
    #     with open(f'./random_generate/asm/test{i}_{len(block)}_nojump.S', 'w') as file:
    #         # file.write("# LLVM-MCA-BEGIN A simple example" + '\n')
    #         for line in block:
    #             file.write(line.code + '\n')
    #         # file.write("# LLVM-MCA-END")


    # print(block)
    # analyzer = DependencyAnalyzer()
    # raw, war, waw = analyzer.analyze(block)
    # print(f"Analysis results: RAW={raw}, WAR={war}, WAW={waw}")
    # analyzer.print_summary()
    # print()
    # analyzer.print_details()

