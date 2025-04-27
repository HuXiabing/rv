import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import torch
from torch import inference_mode
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config
from data import get_dataloader
from models import get_model


def inference(infer_loader, model, device='cuda'):
    model.eval()
    predictions = []

    with inference_mode():
        for batch in tqdm(infer_loader, desc="Inferencing"):
            x = batch['X'].to(device)
            output = model(x)
            predictions.extend(output.cpu().tolist())

    return predictions

def validate_order(predictions, original_samples):

    assert len(original_samples) == len(predictions), \
        f"样本数量不匹配: 输入{len(original_samples)} vs 输出{len(predictions)}"

    print("\n顺序一致性验证（前7个样本）:")
    for i in range(min(7, len(original_samples))):
        print(f"样本{i}: 输入ID={original_samples[i].get('id', 'N/A')} -> 输出={predictions[i]}")


def get_min_prediction_sample(predictions, original_samples):
    # Find the minimum prediction and its index
    min_value = min(predictions)
    min_index = predictions.index(min_value)
    min_sample_id = original_samples[min_index].get('id', 'N/A')

    print(f"\n最小预测值: {min_value}")
    print(f"最小预测值对应的样本ID: {min_sample_id}")
    print(f"该样本在数据集中的索引位置: {min_index}")

    return min_sample_id, min_index, min_value

def main():
    checkpoint_path = Path('experiments/transformer_20250424_190140/checkpoints/model_best.pth')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = Config(**checkpoint.get('config', {}))

    device = torch.device(config.device)
    inference_loader = get_dataloader(
        config.model_type,
        "inference.json",
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state'])

    predictions = inference(inference_loader, model, device)
    print(predictions)

    with open("inference.json", 'r') as f:
        original_samples = json.load(f)

    min_id, min_idx, min_val = get_min_prediction_sample(predictions, original_samples)
    # Validate the order
    validate_order(predictions, original_samples)

    print("\n详细信息:")
    print(f"最小预测值样本 ID: {min_id}")
    print(f"该样本完整信息: {original_samples[min_idx]['instructions']}")

if __name__ == "__main__":
    main()