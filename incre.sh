python fuzzer.py -n 40000
./parallel_process_asm_files.sh
python preprocess.py --mode incremental --cycle_jsons random_generate/output0.json  random_generate/output2.json  random_generate/output4.json  random_generate/output6.json  random_generate/output8.json random_generate/output1.json  random_generate/output3.json  random_generate/output5.json  random_generate/output7.json  random_generate/output9.json --existing_train_json data/train_data1_4_4u_4u_4u_4u_4u_4u2_4u.json --existing_val_json data/val_data.json --train_json data/train_data1_4_4u_4u_4u_4u_4u_4u2_4u_4u1.json --train_samples 446987 --val_samples 173978
python main.py incremental --model_path experiments/incremental_transformer_20250502_152704/checkpoints/model_best.pth --train_data data/train_data1_4_4u_4u_4u_4u_4u_4u2_4u_4u1.json --batch_size 8 --epoch 50
#python main.py train --model_type transformer --train_data data/train_data2.json --val_data data/val_data.json --experiment_name transformer --epoch 70 --batch_size 4
#python main.py train --model_type transformer --train_data data/train_data3.json --val_data data/val_data.json --experiment_name transformer --epoch 70 --batch_size 4
