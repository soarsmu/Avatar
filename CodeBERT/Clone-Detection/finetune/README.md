```
CUDA_VISIBLE_DEVICES=1 python3 finetune.py \
    --do_eval \
    --train_data_file=../data/label_train.txt \
    --eval_data_file=../data/test_sampled.txt \
    --block_size 400 \
    --eval_batch_size 64 \
    --seed 123456
```