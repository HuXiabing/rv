
# 0511
#python main.py train --model_type gnn --train_data data/xiangshan/train_data8k.json --val_data data/xiangshan/val_data.json --experiment_name gnn8k --epoch 70 --batch_size 4
#python main.py train --model_type gnn --train_data data/xiangshan/train_data4k.json --val_data data/xiangshan/val_data.json --experiment_name gnn4k --epoch 70 --batch_size 4
#python main.py train --model_type gnn --train_data data/xiangshan/train_data16k.json --val_data data/xiangshan/val_data.json --experiment_name gnn16k --epoch 70 --batch_size 4
#python main.py train --model_type gnn --train_data data/xiangshan/train_data8.json --val_data data/xiangshan/val_data.json --experiment_name gnn8 --epoch 70 --batch_size 4
python main.py resume --checkpoint experiments/gnn8_20250510_044808/checkpoints/checkpoint_latest.pth --additional_epochs 70
python main.py train --model_type gnn --train_data data/xiangshan/train_data2.json --val_data data/xiangshan/val_data.json --experiment_name gnn2 --epoch 70 --batch_size 4
python main.py train --model_type gnn --train_data data/xiangshan/train_data7.json --val_data data/xiangshan/val_data.json --experiment_name gnn7 --epoch 70 --batch_size 4
python main.py resume --checkpoint experiments/gnn8_20250510_044808/checkpoints/checkpoint_latest.pth --additional_epochs 70
#python main.py train --model_type gnn --train_data data/xiangshan/train_data1.json --val_data data/xiangshan/val_data.json --experiment_name gnn1 --epoch 70 --batch_size 4
#python main.py train --model_type gnn --train_data data/xiangshan/train_data3.json --val_data data/xiangshan/val_data.json --experiment_name gnn3 --epoch 70 --batch_size 4
#python main.py train --model_type gnn --train_data data/xiangshan/train_data4.json --val_data data/xiangshan/val_data.json --experiment_name gnn4 --epoch 70 --batch_size 4
#python main.py train --model_type gnn --train_data data/xiangshan/train_data5.json --val_data data/xiangshan/val_data.json --experiment_name gnn5 --epoch 70 --batch_size 4
#python main.py train --model_type gnn --train_data data/xiangshan/train_data6.json --val_data data/xiangshan/val_data.json --experiment_name gnn6 --epoch 70 --batch_size 4
