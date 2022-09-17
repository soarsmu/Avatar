```
CUDA_VISIBLE_DEVICES=1 python3 distill.py \
    --do_train \
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
