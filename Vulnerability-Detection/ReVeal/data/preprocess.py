import json
import random

# file_name = '../data/chrome_debian_with_slices_ggnn_similar.json'
# data = json.load(open(file_name))

# random.shuffle(data)

# label_train = data[:int(len(data)*0.4)]
# unlabel_train = data[int(len(data)*0.4):int(len(data)*0.8)]
# valid = data[int(len(data)*0.8):int(len(data)*0.9)]
# test = data[int(len(data)*0.9):]

# with open("./label_train.jsonl", "w") as wf:
#     for d in label_train:
#         wf.write(json.dumps(d) + "\n")

# with open("./unlabel_train.jsonl", "w") as wf:
#     for d in unlabel_train:
#         wf.write(json.dumps(d) + "\n")

# with open("./valid.jsonl", "w") as wf:
#     for d in valid:
#         wf.write(json.dumps(d) + "\n")

# with open("./test.jsonl", "w") as wf:
#     for d in test:
#         wf.write(json.dumps(d) + "\n") 

import numpy as np

data = []
with open("unlabel_train.jsonl") as f:
    for line in f:
        data.append(json.loads(line.strip()))

preds = np.load("pred_unlabel_train.npy").tolist()
# print(preds)
new_data = []
for d, p in zip(data, preds):
    d["soft_label"] = p
    new_data.append(d)

with open("soft_unlabel_train.jsonl", "w") as f:
    for d in new_data:
        f.write(json.dumps(d) + "\n")

preds = np.load("pred_unlabel_train.npy").tolist()

print(len(preds))