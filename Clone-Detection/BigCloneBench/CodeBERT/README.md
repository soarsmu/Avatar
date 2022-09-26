To finetune the large model, pls run:
```
python3 finetune.py \
    --do_train \
    --train_data_file=../../../data/clone_search/label_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 2 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee log/finetune.log
```

```
python3 finetune.py \
    --do_eval \
    --train_data_file=../data/label_train.txt \
    --eval_data_file=../data/test_sampled.txt \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 2 \
    --eval_batch_size 100 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
``` 