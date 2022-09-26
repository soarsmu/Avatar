
We encode model hyperparameters in the `surrogate.jsonl` files. To run evaluation using the released 3 MB model:
```
python3 distill.py \
    --do_eval \
    --train_data_file=../../data/unlabel_train.txt \
    --eval_data_file=../../data/test_sampled.txt \
    --model_dir ../checkpoint \
    --size 3 \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 96 \
    --intermediate_size 64 \
    --n_layers 12 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 100 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456
```
If you'd like to train a 3 MB model from scratch, please run:
```
python3 distill.py \
    --do_train \
    --train_data_file=../../data/unlabel_train.txt \
    --eval_data_file=../../data/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 3 \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 96 \
    --intermediate_size 64 \
    --n_layers 12 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 100 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456
```

For the baseline model, we use the model and results from
https://github.com/soarsmu/Compressor/tree/main/CodeBERT/clone_detection.
