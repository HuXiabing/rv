import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
import torch

def set_plot_style():

    sns.set_style("whitegrid")
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20

def plot_instruction_losses(instruction_losses, instruction_counts=None, save_path=None,
                            title="Instruction Type Loss Distribution"):
    """
    Plot the average loss of the five major instruction types.

    plot_instruction_losses(
            instruction_stats["instruction_avg_loss"],
            instruction_stats["instruction_counts"],
            save_path=instr_viz_path,
            title=f"Average Loss by Instruction Type (Epoch {epoch + 1})"
        )

    Args:
        instruction_losses: Mapping of instruction types to losses
        instruction_counts: Mapping of instruction types to occurrence counts (optional)
        save_path: Save path, if None, do not save
        title: Chart title

    Returns:
        Chart
    """
    set_plot_style()

    mapping_dict = torch.load("data/vocab.dump")
    compare_insts = ['slt', 'sltu', 'slti', 'sltiu']
    shifts_arithmetic_logical_insts = ['add', 'addw', 'and', 'sll', 'sllw', 'sra', 'sraw', 'srl', 'srlw', 'sub', 'subw',
                                       'xor', 'addi', 'addiw', 'andi', 'ori', 'slli', 'slliw', 'srai', 'sraiw', 'srli',
                                       'srliw', 'xori']
    mul_div_insts = ['div', 'divu', 'divuw', 'divw', 'mul', 'mulh', 'mulhsu',
                     'mulhu', 'mulw', 'rem', 'remu', 'remuw', 'remw']
    load_insts = ['lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu']
    store_insts = ['sb', 'sd', 'sh', 'sw']

    new_dict = {key: instruction_losses.get(value, 0.0) for key, value in mapping_dict.items()}

    shifts_arithmetic_logical_ratio = sum(new_dict.get(inst, 0.0) for inst in shifts_arithmetic_logical_insts)
    compare_ratio = sum(new_dict.get(inst, 0.0) for inst in compare_insts)
    mul_div_ratio = sum(new_dict.get(inst, 0.0) for inst in mul_div_insts)
    load_ratio = sum(new_dict.get(inst, 0.0) for inst in load_insts)
    store_ratio = sum(new_dict.get(inst, 0.0) for inst in store_insts)

    instr_types = ["shifts_arithmetic_logical_loss", "compare_loss", "mul_div_loss", "load_loss", "store_loss"]
    losses = [shifts_arithmetic_logical_ratio, compare_ratio, mul_div_ratio, load_ratio, store_ratio]

    plt.figure(figsize=(10, 6))
    # plt.bar()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(instr_types, losses, color='skyblue', linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel("Instruction Types")
    ax.set_ylabel("Average Loss")

    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    # plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")

    return fig

def plot_block_length_losses(block_length_losses, block_length_counts=None, save_path=None,
                             title="Basic Block Length Loss Distribution"):
    """
    Plot the average loss for different basic block lengths

    Args:
        block_length_losses: Mapping of basic block lengths to losses
        block_length_counts: Mapping of basic block lengths to occurrence counts (optional)
        save_path: Save path, if None, do not save
        title: Chart title

    Returns:
        Chart object

    plot_block_length_losses(
            block_length_stats["block_length_avg_loss"],
            block_length_stats["block_length_counts"],
            save_path=block_viz_path,
            title=f"Average Loss by Basic Block Length (Epoch {epoch + 1})"
        )
    """
    set_plot_style()

    block_lengths = sorted([int(k) for k in block_length_losses.keys()])
    losses = [block_length_losses[length] for length in block_lengths]

    bins = np.arange(0, max(block_lengths) + 5, 5)
    bin_indices = np.digitize(block_lengths, bins)

    bin_losses = []
    for i in range(1, len(bins)):
        indices = np.where(bin_indices == i)[0]
        if len(indices) > 0:
            bin_losses.append(np.mean([losses[idx] for idx in indices]))
        else:
            bin_losses.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    ax.bar(x_labels, bin_losses, color='lightgreen', edgecolor='black', linewidth=0.5)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Basic Block Length Range', fontsize=14)
    ax.set_ylabel('Average Loss', fontsize=14)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xticks(rotation=45, ha='right')  # 旋转 x 轴标签
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")

    # plt.show()
    return fig

def plot_prediction_scatter(y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           save_path: Optional[str] = None,
                           title: str = "Prediction vs Ground Truth",
                           x_label: str = "Ground Truth",
                           y_label: str = "Prediction") -> plt.Figure:
    """
    绘制预测值与真实值的散点图
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        save_path: 保存路径，如果为None则不保存
        title: 图表标题
        x_label: x轴标签
        y_label: y轴标签
        
    Returns:
        图表对象
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点图
    scatter = ax.scatter(y_true, y_pred, alpha=0.6, c='#3498db', edgecolors='k', linewidths=0.5)
    
    # 添加对角线（完美预测线）
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            'r--', label='Perfect Prediction')
    
    # 计算并显示相关系数
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    ax.annotate(f'Correlation: {correlation:.4f}', 
               xy=(0.05, 0.95), 
               xycoords='axes fraction',
               fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 设置轴标签和标题
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # 调整坐标轴范围
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    
    return fig

def plot_error_histogram(errors: np.ndarray, 
                        save_path: Optional[str] = None,
                        title: str = "Prediction Error Distribution",
                        x_label: str = "Error",
                        bins: int = 30) -> plt.Figure:
    """
    绘制预测误差直方图
    
    Args:
        errors: 误差数组
        save_path: 保存路径，如果为None则不保存
        title: 图表标题
        x_label: x轴标签
        bins: 直方图的箱数
        
    Returns:
        图表对象
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制直方图
    n, bins, patches = ax.hist(errors, bins=bins, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 添加核密度估计
    sns.kdeplot(errors, color='#e74c3c', ax=ax, label='Density')
    
    # 添加均值和中位数线
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    
    ax.axvline(mean_error, color='#3498db', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
    ax.axvline(median_error, color='#9b59b6', linestyle='-.', linewidth=2, label=f'Median: {median_error:.4f}')
    
    # 设置轴标签和标题
    ax.set_xlabel(x_label)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    
    # 添加图例
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    
    return fig

def plot_error_boxplot(errors_by_group: Dict[str, np.ndarray], 
                      save_path: Optional[str] = None,
                      title: str = "Error Distribution by Group",
                      y_label: str = "Error") -> plt.Figure:
    """
    按组绘制误差箱线图
    
    Args:
        errors_by_group: 按组分类的误差字典
        save_path: 保存路径，如果为None则不保存
        title: 图表标题
        y_label: y轴标签
        
    Returns:
        图表对象
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 准备数据
    data = []
    labels = []
    
    for group, errors in errors_by_group.items():
        data.append(errors)
        labels.append(group)
    
    # 绘制箱线图
    box = ax.boxplot(data, patch_artist=True, labels=labels)
    
    # 设置箱线图颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # 设置轴标签和标题
    ax.set_xlabel('Group')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # 旋转x轴标签（如果有很多组）
    if len(labels) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    
    return fig

def plot_feature_importance(feature_names: List[str], 
                           importance_scores: np.ndarray, 
                           save_path: Optional[str] = None,
                           title: str = "Feature Importance",
                           x_label: str = "Importance Score") -> plt.Figure:
    """
    绘制特征重要性条形图
    
    Args:
        feature_names: 特征名称列表
        importance_scores: 重要性分数数组
        save_path: 保存路径，如果为None则不保存
        title: 图表标题
        x_label: x轴标签
        
    Returns:
        图表对象
    """
    set_plot_style()
    
    # 对特征重要性排序
    sorted_idx = np.argsort(importance_scores)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_scores = importance_scores[sorted_idx]
    
    # 设置图表大小，根据特征数量调整
    fig_height = max(6, len(feature_names) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # 设置横向条形图的颜色渐变
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_scores)))
    
    # 绘制横向条形图
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_scores, align='center', color=colors, edgecolor='black', linewidth=0.5)
    
    # 设置轴标签和标题
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    
    # 反转y轴，使最重要的特征在顶部
    ax.invert_yaxis()
    
    # 添加数值标签
    for i, v in enumerate(sorted_scores):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    
    return fig

def plot_confusion_matrix(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          classes: Optional[List[str]] = None,
                          normalize: bool = False, 
                          save_path: Optional[str] = None,
                          title: str = "Confusion Matrix") -> plt.Figure:
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签数组
        y_pred: 预测标签数组
        classes: 类别名称列表
        normalize: 是否归一化
        save_path: 保存路径，如果为None则不保存
        title: 图表标题
        
    Returns:
        图表对象
    """
    set_plot_style()
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 如果需要归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # 设置类别名称
    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热图
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置轴标签和标题
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    
    return fig


def plot_learning_curves(train_losses: List[float],
                         val_losses: List[float],
                         save_path: Optional[str] = None,
                         title: str = "Learning Curves",
                         metric_name: str = "Loss") -> plt.Figure:
    """
    绘制学习曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径，如果为None则不保存
        title: 图表标题
        metric_name: 指标名称

    Returns:
        图表对象
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-', marker='o', label=f'Training {metric_name}')
    ax.plot(epochs, val_losses, 'r-', marker='s', label=f'Validation {metric_name}')

    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True)

    # 添加最低点标记
    best_epoch = val_losses.index(min(val_losses)) + 1
    best_val = min(val_losses)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax.axhline(y=best_val, color='g', linestyle='--', alpha=0.5)
    ax.annotate(f'Best: {best_val:.4f} (Epoch {best_epoch})',
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 1, best_val * 1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")

    return fig


def plot_lr_schedule(learning_rates: List[float],
                     save_path: Optional[str] = None,
                     title: str = "Learning Rate Schedule") -> plt.Figure:
    """
    绘制学习率调度曲线

    Args:
        learning_rates: 学习率列表
        save_path: 保存路径，如果为None则不保存
        title: 图表标题

    Returns:
        图表对象
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(learning_rates) + 1)

    ax.plot(epochs, learning_rates, 'g-', marker='o')
    ax.set_yscale('log')

    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Learning Rate')
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")

    return fig