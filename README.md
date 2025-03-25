**Quick Start**

preprocessing

  ```bash
#  python main.py preprocess --raw_data data/coremark_k230.json --output_dir data
python preprocess.py --mode full --asm_dirs /path/to/asm1 /path/to/asm2 --cycle_dirs /path/to/cycle1 /path/to/cycle2 [--val_ratio 0.2] --train_samples 1000 --val_samples 200
python preprocess.py --mode incremental --asm_dirs /path/to/new_asm --cycle_dirs /path/to/new_cycle [--existing_processed_json data/processed_data.json --existing_train_json data/train_data.json --existing_val_json data/val_data.json]
  
   python preprocess.py --mode full --asm_dirs ../bb/coremark/asm/ ../bb/gzip/asm/ ../bb/sqlite3/asm/ --cycle_dirs ../k230/coremark_result ../k230/gzip_result ../k230/sqlite3_result/
  
  ```

training

```bash
``python main.py train --model_type transformer --train_data data/train_data.json --val_data data/val_data.json --experiment_name transformer_v1 --epoch 50
`````

random bb generation

```bash
python test.py
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
```
python rvmca/main.py -p input_file= ./random_generate/asm/test6_4_nojump.S

incremental training

```bash
python main.py incremental --model_path experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth
```

resume

  ```bash
  python main.py resume --checkpoint experiments/transformer_v1/checkpoints/checkpoint_epoch_10.pth --additional_epochs 20
  ```

[Abolation experiments take llvm-mca as an example via lstm]

exp1: training 1w
python preprocess.py --mode full --asm_dirs ../bb/asm --cycle_dirs ../bb/llvm-mca/ --train_samples 10000 --val_samples 10000 --train_json data/exp1/train_data1w.json --val_json data/exp1/val_data.json
python main.py train --model_type lstm --train_data data/exp1/train_data1w.json --val_data data/exp1/val_data.json --experiment_name lstm_exp1 --epoch 150 --batch_size 64
=====================================

exp2: training 5k + rg 5k
python preprocess.py --mode full --asm_dirs ../bb/asm --cycle_dirs ../bb/llvm-mca/ --train_samples 5000 --val_samples 10000 --train_json data/exp2/train_data5k.json --val_json data/exp2/val_data.json
python main.py train --model_type lstm --train_data data/exp2/train_data5k.json --val_data data/exp2/val_data.json --experiment_name lstm_exp2 --epoch 150 --batch_size 64
-------------------------------------
python fuzzer.py -n 5000
[./run_llvm_mca.sh]
[./asm2bin.sh random_generate/asm/ random_generate/output.json 
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate]
python preprocess.py --mode incremental --asm_dirs random_generate/asm --cycle_dirs random_generate/mca/ --existing_train_json data/exp2/train_data5k.json --existing_val_json data/exp2/val_data.json --train_json data/exp2/train_data5k_5k.json --train_samples 10000
python main.py incremental --model_path experiments/lstm_exp2_20250325_091432/checkpoints/model_best.pth --train_data data/exp2/train_data5k_5k.json --val_data data/exp2/val_data.json --batch_size 64 --epoch 150
=====================================

exp3: training 5k + rg 2.5k * 2
python preprocess.py --mode full --asm_dirs ../bb/asm --cycle_dirs ../bb/llvm-mca/ --train_samples 5000 --val_samples 10000 --train_json data/exp3/train_data5k.json --val_json data/exp3/val_data.json
python main.py train --model_type lstm --train_data data/exp3/train_data5k.json --val_data data/exp3/val_data.json --experiment_name lstm_exp3 --epoch 150 --batch_size 64
-------------------------------------
python fuzzer.py -n 2500
[./run_llvm_mca.sh]
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
python preprocess.py --mode incremental --asm_dirs random_generate/asm --cycle_dirs random_generate/mca/ --existing_train_json data/exp3/train_data5k.json --existing_val_json data/exp3/val_data.json --train_json data/exp3/train_data5k_2.5k.json --train_samples 7500
python main.py incremental --model_path experiments/transformer_exp2_20250324_165943/checkpoints/checkpoint_latest.pth --tra
in_data data/train_data5k_2.5k.json --batch_size 64 --epoch 150
-------------------------------------
python test.py
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
python preprocess.py --mode incremental --asm_dirs random_generate/asm --cycle_dirs random_generate/random_result/ --existing_train_json data/train_data5k_2.5k.json --existing_val_json data/val_data.json --train_json data/train_data5k_2.5k_2.5k.json
python main.py incremental --model_path [experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth] --train_data data/train_data5k_2.5k_2.5k.json
=====================================
exp4: training 2.5k + rg 2.5k * 3
python preprocess.py --mode full --asm_dirs ../bb/asm --cycle_dirs ../bb/llvm-mca/ --train_samples
2500 --val_samples 10000 --train_json data/exp4/train_data2.5k.json --val_json data/exp4/val_data.json
python main.py train --model_type transformer --train_data data/train_data2.5k.json --val_data data/val_data.json --experiment_name transformer_exp4 --epoch 50
-------------------------------------
python test.py
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
python preprocess.py --mode incremental --asm_dirs random_generate/asm --cycle_dirs random_generate/random_result/ --existing_train_json data/train_data2.5k.json --existing_val_json data/val_data.json --train_json data/train_data2.5k_2.5k.json
python main.py incremental --model_path [experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth] --train_data data/train_data2.5k_2.5k.json
-------------------------------------
python test.py
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
python preprocess.py --mode incremental --asm_dirs random_generate/asm --cycle_dirs random_generate/random_result/ --existing_train_json data/train_data2.5k_2.5k.json --existing_val_json data/val_data.json --train_json data/train_data2.5k_2.5k_2.5k.json
python main.py incremental --model_path [experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth] --train_data data/train_data2.5k_2.5k_2.5k.json
-------------------------------------
python test.py
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
python preprocess.py --mode incremental --asm_dirs random_generate/asm --cycle_dirs random_generate/random_result/ --existing_train_json data/train_data2.5k_2.5k_2.5k.json --existing_val_json data/val_data.json --train_json data/train_data2.5k_2.5k_2.5k_2.5k.json
python main.py incremental --model_path [experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth] --train_data data/train_data2.5k_2.5k_2.5k_2.5k.json

=====================================

exp5: rg 2.5k * 4
python test.py   随机生成2500个bb
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
python preprocess.py --mode full --asm_dirs random_generate/asm --cycle_dirs random_generate/random_result/ --train_samples 2500 --val_samples 0 --train_json data/train_data2.5k.json --val_json data/val_data11.json
python main.py train --model_type transformer --train_data data/train_data2.5k.json --val_data data/val_data.json --experiment_name transformer_exp5 --epoch 50
-------------------------------------
python test.py
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
python preprocess.py --mode incremental --asm_dirs random_generate/asm --cycle_dirs random_generate/random_result/ --existing_train_json data/train_data2.5k.json --existing_val_json data/val_data.json --train_json data/train_data2.5k_2.5k.json
python main.py incremental --model_path [experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth] --train_data data/train_data2.5k_2.5k.json
-------------------------------------
python test.py
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
python preprocess.py --mode incremental --asm_dirs random_generate/asm --cycle_dirs random_generate/random_result/ --existing_train_json data/train_data2.5k_2.5k.json --existing_val_json data/val_data.json --train_json data/train_data2.5k_2.5k_2.5k.json
python main.py incremental --model_path [experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth] --train_data data/train_data2.5k_2.5k_2.5k.json
-------------------------------------
python test.py
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
python preprocess.py --mode incremental --asm_dirs random_generate/asm --cycle_dirs random_generate/random_result/ --existing_train_json data/train_data2.5k_2.5k_2.5k.json --existing_val_json data/val_data.json --train_json data/train_data2.5k_2.5k_2.5k_2.5k.json
python main.py incremental --model_path [experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth] --train_data data/train_data2.5k_2.5k_2.5k_2.5k.json