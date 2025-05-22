# 0.3
# fuzzer = generator('experiments/lstm_20250424_232911/statistics/train_loss_stats_epoch_8.json')
#python fuzzer.py -n 40000
#./parallel_process_asm_files.sh
#python preprocess.py --existing_train_json data/u74/train_data3.json --train_json data/u74/train_data3_4.json --train_samples 600000  --existing_val_json data/u74/val_data.json --mode incremental --cycle_jsons random_generate/output0.json  random_generate/output2.json  random_generate/output4.json  random_generate/output6.json  random_generate/output8.json random_generate/output1.json  random_generate/output3.json  random_generate/output5.json  random_generate/output7.json  random_generate/output9.json
#python main.py incremental --model_path experiments/lstm_20250424_232911/checkpoints/model_best.pth --train_data data/u74/train_data3_4.json --batch_size 4 --epoch 50 --val_data data/u74/val_data.json
#python main.py resume --checkpoint experiments/incremental_lstm_20250514_113229/checkpoints/checkpoint_latest.pth --additional_epochs 50  --train_data data/u74/train_data3_4.json --val_data data/u74/val_data.json

#python fuzzer.py -n 40000
#./parallel_process_asm_files.sh
#python preprocess.py --existing_train_json data/u74/train_data3.json --train_json data/u74/train_data3_41.json --train_samples 600000  --existing_val_json data/u74/val_data.json --mode incremental --cycle_jsons random_generate/output0.json  random_generate/output2.json  random_generate/output4.json  random_generate/output6.json  random_generate/output8.json random_generate/output1.json  random_generate/output3.json  random_generate/output5.json  random_generate/output7.json  random_generate/output9.json
python main.py incremental --model_path experiments/lstm_20250424_232911/checkpoints/model_best.pth --train_data data/u74/train_data3_4.json --batch_size 4 --epoch 50 --val_data data/u74/val_data.json
