**Quick Start**

preprocessing
```bash
python preprocess.py --mode full --cycle_jsons random_generate/output0.json  

python preprocess.py --cycle_jsons random_generate/output2.json --existing_train_json data/xiangshan/train_data1.json --existing_val_json data/xiangshan/val_data.json --train_json data/xiangshan/tr.json
```

training
```bash
python main.py train --model_type transformer --train_data data/xiangshan/tr1.json --val_data data/xiangshan/val_data.json --experiment_name transformer --epoch 50
`````


incremental training
```bash
python main.py incremental --model_path experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth
```

resume
```bash
python main.py resume --checkpoint experiments/transformer_v1/checkpoints/checkpoint_epoch_10.pth --additional_epochs 20
```