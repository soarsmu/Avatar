```
CUDA_VISIBLE_DEVICES=4 python3 finetune.py \
    --output_dir=../checkpoints \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../data/label_train.txt \
    --eval_data_file=../data/test_sampled.txt \
    --test_data_file=../data/test_sampled.txt \
    --epoch 4 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```