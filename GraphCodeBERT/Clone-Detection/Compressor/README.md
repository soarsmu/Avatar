```
CUDA_VISIBLE_DEVICES=4 python3 distill.py \
    --output_dir=../checkpoints \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../data/label_train.txt \
    --eval_data_file=../data/test_sampled.txt \
    --test_data_file=../data/test_sampled.txt \
    --epoch 10 \
    --size 3 \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 96 \
    --intermediate_size 64 \
    --n_layers 12 \
    --vocab_size 1000 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```