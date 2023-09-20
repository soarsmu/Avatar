import os
import json
import torch
import random
import numpy as np
import multiprocessing

from tqdm import tqdm
from torch.utils.data import Dataset
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def BPE(texts, vocab_size, file_path, logger):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Lowercase(),
            normalizers.NFKD(),
            normalizers.Strip(),
            normalizers.StripAccents(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        unk_token="<unk>"
    )

    tokenizer.train_from_iterator(texts, trainer)
    folder = "/".join(file_path.split("/")[:-1])
    tokenizer_path = os.path.join(
        folder, "BPE" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)

    return tokenizer

def WordPiece(texts, vocab_size, file_path, logger):
    tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Lowercase(),
            normalizers.NFKD(),
            normalizers.Strip(),
            normalizers.StripAccents(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        unk_token="<unk>"
    )

    tokenizer.train_from_iterator(texts, trainer)
    folder = "/".join(file_path.split("/")[:-1])
    tokenizer_path = os.path.join(
        folder, "WordPiece" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)

    return tokenizer

def Unigram(texts, vocab_size, file_path, logger):
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Lowercase(),
            normalizers.NFKD(),
            normalizers.Strip(),
            normalizers.StripAccents(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    # tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        unk_token="<unk>"
    )

    tokenizer.train_from_iterator(texts, trainer)
    folder = "/".join(file_path.split("/")[:-1])
    tokenizer_path = os.path.join(
        folder, "Unigram" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)

    return tokenizer

def Word(texts, vocab_size, file_path, logger):
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Lowercase(),
            normalizers.NFKD(),
            normalizers.Strip(),
            normalizers.StripAccents(),
        ]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Digits(individual_digits=True)
        ]
    )

    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )

    tokenizer.train_from_iterator(texts, trainer)
    folder = "/".join(file_path.split("/")[:-1])
    tokenizer_path = os.path.join(
        folder, "Word" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)

    return tokenizer


class DistilledDataset(Dataset):
    def __init__(self, tokenizer_type, vocab_size, file_path, max_sequence_length, logger):
        postfix = file_path.split("/")[-1].split(".")[0]
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)

        folder = "/".join(file_path.split("/")[:-1])
        url_to_code = {}
        with open("/".join(file_path.split("/")[:-1])+"/data.jsonl") as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js["idx"]] = js["func"]

        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if "train" in postfix:
                    url1, url2, label, pred = line.split("\t")
                else:
                    url1, url2, label = line.split("\t")
                    pred = -1
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if pred == "0":
                    pred = 0
                elif pred == "1":
                    pred = 1
                else:
                    pred = -1

                if label == "0":
                    label = 0
                elif label == "1":
                    label = 1
                elif label == "-1":
                    label = pred
                    # label = -1

                data.append((url1, url2, label, pred, max_sequence_length, url_to_code))

        tokenizer_path = os.path.join(
            folder, tokenizer_type + "_" + str(vocab_size) + ".json")

        if os.path.exists(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info("Loading vocabulary from file %s", tokenizer_path)
        else:
            texts = []
            for d in data:
                texts.append(" ".join(url_to_code[d[0]].split()))
                texts.append(" ".join(url_to_code[d[1]].split()))
            if tokenizer_type == "BPE":
                tokenizer = BPE(texts, vocab_size, file_path, logger)
            elif tokenizer_type == "WordPiece":
                tokenizer = WordPiece(texts, vocab_size, file_path, logger)
            elif tokenizer_type == "Unigram":
                tokenizer = Unigram(texts, vocab_size, file_path, logger)
            elif tokenizer_type == "Word":
                tokenizer = Word(texts, vocab_size, file_path, logger)

        if "train" in postfix:
            soft_labels = np.load(os.path.join(
                folder, "preds_unlabel_train.npy")).tolist()

        _mp_data = []
        for i, d in enumerate(data):
            lst = list(d)
            lst.append(tokenizer)
            if "train" in postfix:
                lst.append(soft_labels[i])
            else:
                lst.append([0.1, 0.1])
            _mp_data.append(tuple(lst))

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        self.examples = pool.map(
            preprocess, tqdm(_mp_data, total=len(_mp_data)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].pred), torch.tensor(self.examples[i].soft_label)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess(item):
    url1, url2, label, pred, max_sequence_length, url_to_code, tokenizer, s = item
    code1 = " ".join(url_to_code[url1].split())
    code2 = " ".join(url_to_code[url2].split())
    code1_ids = tokenizer.encode(code1).ids[:max_sequence_length-2]
    code2_ids = tokenizer.encode(code2).ids[:max_sequence_length-2]
    code1_ids = [tokenizer.token_to_id(
        "<s>")]+code1_ids+[tokenizer.token_to_id("</s>")]
    code2_ids = [tokenizer.token_to_id(
        "<s>")]+code2_ids+[tokenizer.token_to_id("</s>")]
    padding_length = max_sequence_length - len(code1_ids)
    code1_ids += [tokenizer.token_to_id("<pad>")] * padding_length
    padding_length = max_sequence_length - len(code2_ids)
    code2_ids += [tokenizer.token_to_id("<pad>")] * padding_length

    source_tokens = code1 + code2
    source_ids = code1_ids + code2_ids

    return InputFeatures(source_tokens, source_ids, label, pred, s)

class InputFeatures(object):
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 pred=0,
                 soft_label=[0.1, 0.1]
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.pred = pred
        self.soft_label = soft_label


# deprecated this class
class Token_Encoder(object):
    def __init__(self, vocab_size, encoding, file_path, logger):
        self.vocab_size = vocab_size
        self.logger = logger
        self.file_path = file_path
        self.encoding = encoding
        folder = "/".join(file_path.split("/")[:-1])
        try:
            tokenizer_path = os.path.join(
                folder, "token_encoder", encoding + "_" + str(vocab_size) + ".json")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.logger.info("Loading vocabulary from file %s", tokenizer_path)
        except:
            data = []
            with open(self.file_path) as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            texts = [" ".join(d["func"].split()) for d in data]

            if encoding == "token":
                self.tokenizer = self.token(texts)
            elif encoding == "subtoken":
                self.tokenizer = self.subtoken(texts)
            elif encoding == "BPE":
                self.tokenizer = self.BPE(texts)

    def token(self, texts):
        tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.pre_tokenizer = Whitespace()
        trainer = trainers.WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=[
                                            "<s>", "<pad>", "</s>", "<unk>"])
        tokenizer.train_from_iterator(texts, trainer)
        folder = "/".join(self.file_path.split("/")[:-1])
        os.makedirs(os.path.join(folder, "token_encoder"), exist_ok=True)
        tokenizer_path = os.path.join(
            folder, "token_encoder", self.encoding + "_" + str(self.vocab_size) + ".json")
        tokenizer.save(tokenizer_path, pretty=True)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer

    def subtoken(self, texts):
        tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=[
                                   "<s>", "<pad>", "</s>", "<unk>"])
        tokenizer.train_from_iterator(texts, trainer)
        folder = "/".join(self.file_path.split("/")[:-1])
        tokenizer_path = os.path.join(
            folder, "token_encoder", self.encoding + "_" + str(self.vocab_size) + ".json")
        os.makedirs(os.path.join(folder, "token_encoder"), exist_ok=True)
        tokenizer.save(tokenizer_path, pretty=True)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer

    def BPE(self, texts):
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
        )

        tokenizer.train_from_iterator(texts, trainer)
        folder = "/".join(self.file_path.split("/")[:-1])
        tokenizer_path = os.path.join(
            folder, "token_encoder", self.encoding + "_" + str(self.vocab_size) + ".json")
        os.makedirs(os.path.join(folder, "token_encoder"), exist_ok=True)
        tokenizer.save(tokenizer_path, pretty=True)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer
