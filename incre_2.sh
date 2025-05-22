# 0.2
#    fuzzer = generator('experiments/lstm_20250427_210029/statistics/train_loss_stats_epoch_8.json')
#    fuzzer = incre_generator('experiments/incremental_lstm_20250513_082153/statistics/train_loss_stats_epoch_22.json', 0.064751, fuzzer)
#    fuzzer = incre_generator('experiments/incremental_lstm_20250515_230133/statistics/train_loss_stats_epoch_11.json', 0.062639, fuzzer)


#python fuzzer.py -n 40000
#./parallel_process_asm_files.sh
#python preprocess.py --train_json data/u74/train_data2_4_4.json --train_samples 450000 --existing_train_json data/u74/train_data2_4.json --mode incremental --existing_val_json data/u74/val_data.json  --cycle_jsons random_generate/output0.json  random_generate/output2.json  random_generate/output4.json  random_generate/output6.json  random_generate/output8.json random_generate/output1.json  random_generate/output3.json  random_generate/output5.json  random_generate/output7.json  random_generate/output9.json
#python main.py incremental --train_data data/u74/train_data2_4_4.json --model_path experiments/incremental_lstm_20250513_082153/checkpoints/model_best.pth  --batch_size 4 --epoch 50 --val_data data/u74/val_data.json
#python main.py resume --checkpoint experiments/incremental_lstm_20250514_235351/checkpoints/checkpoint_latest.pth --additional_epochs 50  --train_data data/u74/train_data2_4_4.json --val_data data/u74/val_data.json


#2025-05-16 01:17:22,430 - incremental_lstm_20250515_230133 - INFO - Epoch 10 - val_metrics: loss: 0.062639, accuracy25: 0.959041, accuracy10: 0.825857, accuracy5: 0.658123, is_best_accuracy: 1.000000
python fuzzer.py -n 40000
./parallel_process_asm_files.sh
python preprocess.py --train_json data/u74/train_data2_4_4_4.json --train_samples 600000 --existing_train_json data/u74/train_data2_4_4.json --mode incremental --existing_val_json data/u74/val_data.json  --cycle_jsons random_generate/output0.json  random_generate/output2.json  random_generate/output4.json  random_generate/output6.json  random_generate/output8.json random_generate/output1.json  random_generate/output3.json  random_generate/output5.json  random_generate/output7.json  random_generate/output9.json
python main.py incremental --train_data data/u74/train_data2_4_4_4.json --model_path experiments/incremental_lstm_20250515_230133/checkpoints/model_best.pth  --batch_size 4 --epoch 50 --val_data data/u74/val_data.json

