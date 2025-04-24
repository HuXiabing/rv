#!/usr/bin/env python
"""
恢复训练脚本 - 从检查点恢复RISC-V吞吐量预测模型训练
"""

import os
import sys
import argparse
from pathlib import Path

from torch import inference_mode

sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from config import Config
from data import get_dataloader
from models import get_model
from trainers import Trainer
from utils import set_seed, ExperimentManager


def main():
    checkpoint_path = Path('experiments/starfive_lstm_exp1_20250418_091715/checkpoints/model_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config_dict = checkpoint.get('config', None)
    if config_dict is None:
        raise ValueError(f"There is no configuration information in {checkpoint_path}")

    config = Config(**config_dict)

    device = torch.device(config.device)
    print(device)
    config.checkpoint = checkpoint_path

    inference_data = torch.tensor([[75,5,25,6,18,7,4,0],[137,5,21,6,25,31,4,0],[108,5,23,6,32,18,4,0],[74,5,24,6,9,7,4,0],[74,5,22,6,9,7,4,0],[74,5,20,6,35,7,4,0],[74,5,19,6,29,7,4,0],[74,5,27,6,18,7,4,0],[74,5,18,6,25,7,4,0],[75,5,21,6,21,7,4,0],[79,5,10,6,7,4,0,0]]).to(device)
    x = torch.zeros((64, 8), dtype=torch.long)
    x[:inference_data.shape[0]] = inference_data
    # print(x.unsqueeze(0))

    model = get_model(config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    print("Model device:", next(model.parameters()).device)

    model.eval()
    with torch.no_grad():
        output = model(x.unsqueeze(0).to(device))
        print(output)


if __name__ == "__main__":
    main()
