**Quick Start**

preprocessing

  ```bash
  python main.py preprocess --raw_data data/coremark_k230.json --output_dir data
  ```

training

  ```bash
  python main.py train --model_type transformer --train_data data/train_data.h5 --val_data data/val_data.h5 --experiment_name transformer_v1 --epoch 50
  ```

random bb generation

```bash
python test.py
./asm2bin.sh random_generate/asm/ random_generate/binary
scp -r random_generate/binary/ root@192.168.0.110:~/run
wait until board is .................
scp -r root@192.168.0.110:~/run/random_result ./random_generate
scp -r root@192.168.0.110:~/run/random_result.txt ./random_generate
python json_gen.py
```

incremental preprocessing

```bash
python main.py incremental_preprocess --raw_data data/output.json --output_dir data
```

incremental training

```bash
python main.py incremental --model_path experiments/transformer_v1_20250304_101004/checkpoints/checkpoint_latest.pth
```



evaluation

not tested yet

  ```bash
  python main.py evaluate --model_path experiments/transformer_v1/checkpoints/model_best.pth --test_data data/test_data.h5 --output_dir evaluation/transformer_v1
  ```

inference

not tested yet

  ```bash
  python main.py predict --model_path experiments/transformer_v1/checkpoints/model_best.pth --input_json data/new_samples.json --output predictions.json
  ```

resume

not tested yet

  ```bash
  python main.py resume --checkpoint experiments/transformer_v1/checkpoints/checkpoint_epoch_10.pth --additional_epochs 20
  ```

  
