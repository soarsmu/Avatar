import csv
import random
import logging

from sklearn import linear_model, neural_network
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from distill_utils import distill


def uniform_sampling(space, num_samples):
    space_size = len(space)
    samples = []

    for _ in range(num_samples):
        random_index = random.randint(0, space_size - 1)
        samples.append(space[random_index])

    return samples

def get_sampled_data(spaces, num_samples):
    sampled_data = []
    for space in spaces:
        space = [*range(space[0], space[1] + 1)]
        sampled_data.append(uniform_sampling(space, num_samples))

    combined_data = []

    for i in range(len(sampled_data[0])):
        combined_data.append([sampled_data[j][i] for j in range(len(sampled_data))])

    for data in combined_data:
        while data[0] % data[-1] != 0:
            space = spaces[0]
            space = [*range(space[0], space[1] + 1)]
            data[0] = uniform_sampling(space, 1)[0]

    return combined_data

def get_train_data(search_space, num_samples):
    sampled_data = get_sampled_data(search_space, num_samples)

    accs = distill(sampled_data, surrogate=True)
    return [sampled_data, accs]

def predictor(dataset):
    X_train, y_train = dataset[0], dataset[1]
    # X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2)
    reg = linear_model.BayesianRidge()
    reg.fit(X_train, y_train)

    # preds = reg.predict(X_test)
    # logging.info("MAE: {}".format(mean_absolute_error(y_test, preds)))

    return reg

def reverse_hyperparams_convert(hyperparams):
    tokenizer_type = {"BPE": 1, "WordPiece": 2, "Unigram": 3, "Word": 4}
    hidden_act = {"gelu": 1, "relu": 2, "silu": 3, "gelu_new": 4}
    position_embedding_type = {"absolute": 1, "relative_key": 2, "relative_key_query": 3}
    learning_rate = {"0.001": 1, "0.0001": 2, "5e-05": 3}
    batch_size = {"8": 1, "16": 2}

    return [
        tokenizer_type[hyperparams[0]],
        hyperparams[1],
        hyperparams[2],
        hyperparams[3],
        hidden_act[hyperparams[4]],
        hyperparams[5],
        hyperparams[6],
        hyperparams[7],
        hyperparams[8],
        hyperparams[9],
        position_embedding_type[hyperparams[10]],
        learning_rate[hyperparams[11]],
        batch_size[hyperparams[12]]
    ], float(hyperparams[13])


if __name__ == "__main__":
    train_data = []
    train_accs = []
    with open("surrogate_train_data.csv") as f:
        reader = csv.reader(f) 
        for i, row in enumerate(reader): 
            if i == 0:
                continue
            row = reverse_hyperparams_convert(row)
            train_data.append([float(x) for x in row[0]])
            train_accs.append(row[1])
    
    reg = predictor([train_data, train_accs])
            
            