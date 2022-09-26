
To run evaluation using the released 3 MB model:
```
python3 distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_test \
    --train_data_file=../../data/unlabel_train.txt \
    --eval_data_file=../../data/valid_sampled.txt \
    --test_data_file=../../data/test_sampled.txt \
    --epoch 10 \
    --size 3 \
    --type unlabel_train \
    --attention_heads 2 \
    --hidden_dim 24 \
    --intermediate_size 1508 \
    --n_layers 1 \
    --vocab_size 27505 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 12 \
    --eval_batch_size 100 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```
If you'd like to train a 3 MB model from scratch, please run:
```
python3 distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../data/unlabel_train.txt \
    --eval_data_file=../../data/valid_sampled.txt \
    --test_data_file=../../data/test_sampled.txt \
    --epoch 10 \
    --size 3 \
    --type unlabel_train \
    --attention_heads 2 \
    --hidden_dim 24 \
    --intermediate_size 1508 \
    --n_layers 1 \
    --vocab_size 27505 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 12 \
    --eval_batch_size 100 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

For the baseline model, we use the model and results from
https://github.com/soarsmu/Compressor/tree/main/GraphCodeBERT/clone_detection.