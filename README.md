**Quick Start**

preprocessing

  ```bash
#  python main.py preprocess --raw_data data/coremark_k230.json --output_dir data
python preprocess.py --mode full --asm_dirs /path/to/asm1 /path/to/asm2 --cycle_dirs /path/to/cycle1 /path/to/cycle2 [--val_ratio 0.2]
python preprocess.py --mode incremental --asm_dirs /path/to/new_asm --cycle_dirs /path/to/new_cycle [--existing_processed_json data/processed_data.json --existing_train_json data/train_data.json --existing_val_json data/val_data.json]
  
   python preprocess.py --mode full --asm_dirs ../bb/coremark/asm/ ../bb/gzip/asm/ ../bb/sqlite3/asm/ --cycle_dirs ../k230/coremark_result ../k230/gzip_result ../k230/sqlite3_result/
  
  ```

training

```bash
``python main.py train --model_type transformer --train_data data/train_data.h5 --val_data data/val_data.h5 --experiment_name transformer_v1 --epoch 50
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


incremental training

```bash
python main.py incremental --model_path experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth
```

resume

  ```bash
  python main.py resume --checkpoint experiments/transformer_v1/checkpoints/checkpoint_epoch_10.pth --additional_epochs 20
  ```




  
