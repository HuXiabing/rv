**Quick Start**

preprocessing

  ```bash
#  python main.py preprocess --raw_data data/coremark_k230.json --output_dir data
python preprocess.py --mode full --cycle_jsons ../u74/nanhu_0.json  ../u74/nanhu_4.json  ../u74/nanhu_8.json
../u74/nanhu_1.json  ../u74/nanhu_5.json  ../u74/nanhu_9.json
../u74/nanhu_2.json  ../u74/nanhu_6.json
../u74/nanhu_3.json  ../u74/nanhu_7.json
 --train_samples 4000 --val_samples 4000 --train_json data/exp4/train_data4k.json --val_json data/exp4/val_data.json
python preprocess.py --mode incremental --cycle_jsons random_generate/mca.json --existing_train_json data/exp4/train_data4k.json --existing_val_json data/exp4/val_data.json --train_json data/exp4/train_data4k_4k.json --train_samples 8000 --val_samples 4000
  ```

training

```bash
``python main.py train --model_type transformer --train_data data/train_data.json --val_data data/val_data.json --experiment_name transformer_v1 --epoch 50
`````

python rvmca/main.py -p input_file= ./random_generate/asm/test6_4_nojump.S

incremental training

```bash
python main.py incremental --model_path experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth
```

resume

  ```bash
  python main.py resume --checkpoint experiments/transformer_v1/checkpoints/checkpoint_epoch_10.pth --additional_epochs 20
  ```