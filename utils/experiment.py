import os
import json
import time
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

class ExperimentManager:
    """实验管理器，负责跟踪和记录实验"""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        """
        初始化实验管理器
        
        Args:
            experiment_name: 实验名称
            base_dir: 实验基础目录
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        
        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        
        # 设置实验目录
        self.experiment_dir = os.path.join(base_dir, self.experiment_id)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.output_dir = os.path.join(self.experiment_dir, "outputs")
        
        # 创建必要的目录
        self.setup_directories()
        
        # 设置日志
        self.setup_logger()
        
        # 实验指标
        self.metrics = {}
        self.history = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
            "metrics": {}
        }
        
        # 记录开始时间
        self.start_time = time.time()
        self.logger.info(f"创建实验: {self.experiment_id}")
    
    def setup_directories(self):
        """创建实验相关目录"""
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.INFO)
        
        # 确保处理程序不会重复添加
        if not self.logger.handlers:
            # 文件处理程序
            log_file = os.path.join(self.log_dir, "experiment.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # 控制台处理程序
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建格式化器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加处理程序
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def save_config(self, config):
        """
        保存实验配置
        
        Args:
            config: 配置对象
        """
        config_path = os.path.join(self.experiment_dir, "config.json")
        
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = dict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        self.logger.info(f"配置保存到 {config_path}")

    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """
        记录训练/验证指标

        Args:
            metrics: 指标字典
            step: 当前步数（如epoch）
            prefix: 指标前缀（如'train_'或'val_'）
        """
        # 记录到实验指标
        if prefix not in self.metrics:
            self.metrics[prefix] = {}

        for name, value in metrics.items():
            metric_name = f"{prefix}{name}"

            if metric_name not in self.metrics[prefix]:
                self.metrics[prefix][name] = []

            self.metrics[prefix][name].append((step, value))

            # 记录到历史
            if metric_name not in self.history["metrics"]:
                self.history["metrics"][metric_name] = []

            self.history["metrics"][metric_name].append(value)

        # 构建指标字符串，处理不同类型的值
        metrics_str_parts = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                # 如果是数值类型，使用.6f格式
                metrics_str_parts.append(f"{name}: {value:.6f}")
            else:
                # 其他类型则直接转为字符串
                metrics_str_parts.append(f"{name}: {value}")

        metrics_str = ", ".join(metrics_str_parts)

        # 记录到日志
        self.logger.info(f"Step {step} - {prefix} metrics: {metrics_str}")

        # 保存指标
        self.save_metrics()

    
    def log_train_val_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int):
        """
        记录训练和验证指标
        
        Args:
            train_metrics: 训练指标字典
            val_metrics: 验证指标字典
            epoch: 当前周期
        """
        # 记录训练指标
        self.log_metrics(train_metrics, epoch, prefix="train_")
        
        # 记录验证指标
        self.log_metrics(val_metrics, epoch, prefix="val_")
        
        # 更新历史
        if "loss" in train_metrics:
            self.history["train_losses"].append(train_metrics["loss"])
        if "loss" in val_metrics:
            self.history["val_losses"].append(val_metrics["loss"])
        
        # 保存历史
        self.save_history()
    
    def log_learning_rate(self, lr: float, epoch: int):
        """
        记录学习率
        
        Args:
            lr: 学习率
            epoch: 当前周期
        """
        self.history["learning_rates"].append(lr)
        self.logger.info(f"Epoch {epoch} - Learning rate: {lr:.8f}")
        
        # 保存历史
        self.save_history()
    
    def save_metrics(self):
        """保存指标到文件"""
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def save_history(self):
        """保存训练历史到文件"""
        history_path = os.path.join(self.log_dir, "history.json")
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def log_artifacts(self, artifacts_dict: Dict[str, str]):
        """
        保存实验相关文件
        
        Args:
            artifacts_dict: 文件名到路径的字典
        """
        artifacts_dir = os.path.join(self.experiment_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        for name, path in artifacts_dict.items():
            if os.path.exists(path):
                dest_path = os.path.join(artifacts_dir, name)
                shutil.copy(path, dest_path)
                self.logger.info(f"保存文件 {name} 到 {dest_path}")
            else:
                self.logger.warning(f"找不到文件: {path}")
    
    def save_summary(self, summary_data: Dict[str, Any]):
        """
        保存实验摘要信息
        
        Args:
            summary_data: 摘要数据字典
        """
        # 添加时间信息
        summary_data['start_time'] = self.timestamp
        summary_data['duration'] = time.time() - self.start_time
        
        summary_path = os.path.join(self.experiment_dir, "summary.json")
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        self.logger.info(f"实验摘要保存到 {summary_path}")
    
    def get_checkpoint_dir(self) -> str:
        """
        获取检查点保存目录
        
        Returns:
            检查点目录路径
        """
        return self.checkpoint_dir
    
    def get_output_dir(self) -> str:
        """
        获取输出文件目录
        
        Returns:
            输出目录路径
        """
        return self.output_dir
    
    def finish(self):
        """完成实验，记录总结信息"""
        # 计算持续时间
        duration = time.time() - self.start_time
        
        # 记录结束消息
        self.logger.info(f"实验完成. 总时间: {duration:.2f} 秒")
        
        # 保存总结
        self.save_summary({
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'duration': duration,
            'metrics': self.metrics
        })

    def save_instruction_stats(self, instruction_stats, epoch):
        """
        保存指令类型统计信息

        Args:
            instruction_stats: 指令统计信息字典
            epoch: 当前周期
        """
        stats_dir = os.path.join(self.experiment_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)

        stats_path = os.path.join(stats_dir, f"instruction_stats_epoch_{epoch + 1}.json")

        with open(stats_path, 'w') as f:
            json.dump(instruction_stats, f, indent=4)

        self.logger.info(f"指令统计信息已保存到 {stats_path}")

    def save_block_length_stats(self, block_length_stats, epoch):
        """
        保存基本块长度统计信息

        Args:
            block_length_stats: 基本块长度统计信息字典
            epoch: 当前周期
        """
        stats_dir = os.path.join(self.experiment_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)

        stats_path = os.path.join(stats_dir, f"block_length_stats_epoch_{epoch + 1}.json")

        with open(stats_path, 'w') as f:
            json.dump(block_length_stats, f, indent=4)

        self.logger.info(f"基本块长度统计信息已保存到 {stats_path}")

    # def visualize_epoch_stats(self, instruction_stats, block_length_stats, epoch):
    #     """
    #     为当前周期生成统计可视化
    #
    #     Args:
    #         instruction_stats: 指令统计信息字典
    #         block_length_stats: 基本块长度统计信息字典
    #         epoch: 当前周期
    #     """
    #     from utils.visualize import plot_instruction_losses, plot_block_length_losses
    #
    #     viz_dir = os.path.join(self.experiment_dir, "visualizations")
    #     os.makedirs(viz_dir, exist_ok=True)
    #
    #     # 绘制指令类型损失分布
    #     instr_viz_path = os.path.join(viz_dir, f"instruction_losses_epoch_{epoch + 1}.png")
    #     plot_instruction_losses(
    #         instruction_stats["instruction_avg_loss"],
    #         instruction_stats["instruction_counts"],
    #         save_path=instr_viz_path,
    #         title=f"Average Loss by Instruction Type (Epoch {epoch + 1})"
    #     )
    #
    #     # 绘制基本块长度损失分布
    #     block_viz_path = os.path.join(viz_dir, f"block_length_losses_epoch_{epoch + 1}.png")
    #     plot_block_length_losses(
    #         block_length_stats["block_length_avg_loss"],
    #         block_length_stats["block_length_counts"],
    #         save_path=block_viz_path,
    #         title=f"Average Loss by Basic Block Length (Epoch {epoch + 1})"
    #     )
    #
    #     self.logger.info(f"已为周期 {epoch + 1} 生成统计可视化")

    def visualize_epoch_stats(self, instruction_stats, block_length_stats, epoch):
        """
        为当前周期生成统计可视化

        Args:
            instruction_stats: 指令统计信息字典
            block_length_stats: 基本块长度统计信息字典
            epoch: 当前周期
        """
        from utils.visualize import plot_instruction_losses, plot_block_length_losses
        from utils.analysis import analyze_instruction_statistics, analyze_block_length_statistics

        viz_dir = os.path.join(self.experiment_dir, "visualizations")
        stats_dir = os.path.join(self.experiment_dir, "stats")
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        # 保存原始统计数据
        instruction_avg_loss_path = os.path.join(stats_dir, f"instruction_avg_loss_epoch_{epoch + 1}.txt")
        block_length_avg_loss_path = os.path.join(stats_dir, f"block_length_avg_loss_epoch_{epoch + 1}.txt")

        with open(instruction_avg_loss_path, 'w') as file:
            for key, value in instruction_stats["instruction_avg_loss"].items():
                file.write(f'{key}: {value}\n')

        with open(block_length_avg_loss_path, 'w') as file:
            for key, value in block_length_stats["block_length_avg_loss"].items():
                file.write(f'{key}: {value}\n')

        # 分析统计数据
        analysis_output_dir = os.path.join(self.experiment_dir, f"analysis_epoch_{epoch + 1}")

        instruction_vec = analyze_instruction_statistics(
            instruction_stats["instruction_avg_loss"],
            mapping_dict_path="data/mapping_dict.dump",
            output_dir=analysis_output_dir
        )

        block_dict = analyze_block_length_statistics(
            block_length_stats["block_length_avg_loss"],
            output_dir=analysis_output_dir
        )

        # 保存分析结果
        analysis_summary = {
            "instruction_vec": instruction_vec,
            "block_dict": {str(k): v for k, v in block_dict.items()}  # 转换键为字符串以便JSON序列化
        }

        with open(os.path.join(analysis_output_dir, "analysis_summary.json"), 'w') as f:
            json.dump(analysis_summary, f, indent=2)

        # 绘制指令类型损失分布
        instr_viz_path = os.path.join(viz_dir, f"instruction_losses_epoch_{epoch + 1}.png")
        plot_instruction_losses(
            instruction_stats["instruction_avg_loss"],
            instruction_stats["instruction_counts"],
            save_path=instr_viz_path,
            title=f"Average Loss by Instruction Type (Epoch {epoch + 1})"
        )

        # 绘制基本块长度损失分布
        block_viz_path = os.path.join(viz_dir, f"block_length_losses_epoch_{epoch + 1}.png")
        plot_block_length_losses(
            block_length_stats["block_length_avg_loss"],
            block_length_stats["block_length_counts"],
            save_path=block_viz_path,
            title=f"Average Loss by Basic Block Length (Epoch {epoch + 1})"
        )

        self.logger.info(f"已为周期 {epoch + 1} 生成统计可视化和分析")
