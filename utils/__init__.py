# from .metrics import compute_regression_metrics, compute_error_distribution
from .visualize import (
    plot_learning_curves, 
    plot_lr_schedule, 
    plot_prediction_scatter,
    plot_error_histogram,
    plot_error_boxplot,
    plot_feature_importance,
    plot_confusion_matrix
)
from .experiment import ExperimentManager
from .seed import set_seed

__all__ = [
    # 可视化工具
    'plot_learning_curves',
    'plot_lr_schedule',
    'plot_prediction_scatter',
    'plot_error_histogram',
    'plot_error_boxplot',
    'plot_feature_importance',
    'plot_confusion_matrix',
    # 实验管理
    'ExperimentManager',
    # 随机种子
    'set_seed',

]
