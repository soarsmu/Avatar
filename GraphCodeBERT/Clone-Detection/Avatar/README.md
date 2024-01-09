### Run the search process

Please run the following command to search for the best configurations for the model.
```
CUDA_VISIBLE_DEVICES=1 python3 MOO.py 2>&1| tee ../logs/search.log
```
After the search process is finished, the best configurations will be saved in `pareto_set.csv`.


### Run the training process

Please select a group of configurations from `pareto_set.csv`, such as the one with the 3 MB model size. Then, please modify the `distill_utils.py` file to use the selected configurations and set `eval=False, surrogate=False`, and run the following command to train the model.
```
CUDA_VISIBLE_DEVICES=1 python3 distill_utils 2>&1| tee ../logs/train.log
```

If you want to evaluate the model checkpoint that we provide but not train it, please update the `distill_utils.py` file with setting the `eval=True, surrogate=False` and run the above command.