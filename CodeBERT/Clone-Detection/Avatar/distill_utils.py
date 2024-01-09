import os
import time
import torch
import logging
import warnings
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from models import Model, distill_loss
from utils import set_seed, DistilledDataset
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel


warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

def train(model, train_dataloader, eval_dataloader, epochs, learning_rate, device, surrogate=False):
    num_steps = len(train_dataloader) * epochs
    no_decay = ["bias", "LayerNorm.weight"]
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{total_params:,} total parameters.")
    logger.info(f"{total_params*4/1e6} MB model size")
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_steps*0.1,
                                                num_training_steps=num_steps)
    dev_best_acc = 0

    for epoch in range(epochs):
        model.train()
        tr_num = 0
        train_loss = 0

        logger.info("Epoch [{}/{}]".format(epoch + 1, epochs))
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        bar.set_description("Train")
        for batch in bar:
            texts = batch[0].to(device)
            soft_knowledge = batch[3].to(device)
            preds = model(texts)
            loss = distill_loss(preds, soft_knowledge)

            loss.backward()
            train_loss += loss.item()
            tr_num += 1

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dev_results = evaluate(model, device,  eval_dataloader)
        dev_acc = dev_results["eval_f1"]
        if dev_acc >= dev_best_acc:
            dev_best_acc = dev_acc
            if not surrogate:
                output_dir = os.path.join("../checkpoints", "Avatar")
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_dir, "model.bin"))
                logger.info("New best model found and saved.")

        logger.info("Train Loss: {0}, Val Acc: {1}, Val Precision: {2}, Val Recall: {3}, Val F1: {4}".format(train_loss/tr_num, dev_results["eval_acc"], dev_results["eval_precision"], dev_results["eval_recall"], dev_results["eval_f1"]))

    return dev_best_acc

def evaluate(model, device, eval_dataloader):
    model.eval()
    predict_all = []
    labels_all = []
    time_count = []
    with torch.no_grad():
        bar = tqdm(eval_dataloader, total=len(eval_dataloader))
        bar.set_description("Evaluation")
        for batch in bar:
            texts = batch[0].to(device)
            label = batch[1].to(device)
            time_start = time.time()
            prob = model(texts)
            time_end = time.time()
            prob = F.softmax(prob)
            predict_all.append(prob.cpu().numpy())
            labels_all.append(label.cpu().numpy())
            time_count.append(time_end-time_start)

    latency = np.mean(time_count)
    logger.info("Average Inference Time pre Batch: {}".format(latency))
    predict_all = np.concatenate(predict_all, 0)
    labels_all = np.concatenate(labels_all, 0)

    preds = predict_all[:, 1] > 0.5
    recall = recall_score(labels_all, preds)
    precision = precision_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    results = {
        "eval_acc": np.mean(labels_all==preds),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
        "inference_time": latency
    }
    return results

def distill(hyperparams_set, eval=False, surrogate=True):

    train_data_file = "../data/unlabel_train.txt"
    eval_data_file = "../data/valid_sampled.txt"
    test_data_file = "../data/test_sampled.txt"
    seed = 123456
    if surrogate:
        epochs = 5
    else:
        epochs = 10
    n_labels = 2
    device = torch.device("cuda")

    set_seed(seed)

    dev_best_accs = []
    for hyperparams in hyperparams_set:
        tokenizer_type, vocab_size, num_hidden_layers, hidden_size, hidden_act, hidden_dropout_prob, intermediate_size, num_attention_heads, attention_probs_dropout_prob, max_sequence_length, position_embedding_type, learning_rate, batch_size = hyperparams_convert(hyperparams)

        config = RobertaConfig.from_pretrained("microsoft/codebert-base")
        config.num_labels = n_labels
        config.vocab_size = vocab_size
        config.num_hidden_layers = num_hidden_layers
        config.hidden_size = hidden_size
        config.hidden_act = hidden_act
        config.hidden_dropout_prob = hidden_dropout_prob
        config.intermediate_size = intermediate_size
        config.num_attention_heads = num_attention_heads
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.max_position_embeddings = max_sequence_length+2
        config.position_embedding_type = position_embedding_type

        model = Model(RobertaModel(config=config), config)
        if not eval:
            train_dataset = DistilledDataset(tokenizer_type, vocab_size, train_data_file, max_sequence_length, logger)
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=8, pin_memory=True)

            eval_dataset = DistilledDataset(tokenizer_type, vocab_size, eval_data_file, max_sequence_length, logger)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size*2, num_workers=8, pin_memory=True)

            model.to(device)
            dev_best_acc = train(model, train_dataloader, eval_dataloader, epochs, learning_rate, device, surrogate)
            dev_best_accs.append(dev_best_acc)
        else:
            model_dir = os.path.join("../checkpoints", "Avatar", "model.bin")
            model.load_state_dict(torch.load(model_dir))
            test_dataset = DistilledDataset(tokenizer_type, vocab_size, test_data_file, max_sequence_length, logger)
            test_sampler = SequentialSampler(test_dataset)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size*2, num_workers=8, pin_memory=True)

            model.to(device)
            test_results = evaluate(model, device, test_dataloader)
            logger.info("Test Acc: {0}, Test Precision: {1}, Test Recall: {2}, Test F1: {3}".format(test_results["eval_acc"], test_results["eval_precision"], test_results["eval_recall"], test_results["eval_f1"], test_results["inference_time"]))

    return dev_best_accs

def hyperparams_convert(hyperparams):
    tokenizer_type = {1: "BPE", 2: "WordPiece", 3: "Unigram", 4: "Word"}
    hidden_act = {1: "gelu", 2: "relu", 3: "silu", 4: "gelu_new"}
    position_embedding_type = {1: "absolute", 2: "relative_key", 3: "relative_key_query"}
    learning_rate = {1: 1e-3, 2: 1e-4, 3: 5e-5}
    batch_size = {1: 8, 2: 16}

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
    ]


if __name__ == "__main__":
    print(hyperparams_convert([1,27505,1,24,3,0.2,1508,2,0.1,512,1,2,2]))
    distill([[1,27505,3,36,3,0.3,1508,12,0.2,358,1,2,2]], eval=True, surrogate=False)

