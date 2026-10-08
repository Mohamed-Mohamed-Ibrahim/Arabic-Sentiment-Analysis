import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchtext.vocab import build_vocab_from_iterator
from transformers import AutoTokenizer
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

RAW_DATA_PATH = "data/MSA Dataset.csv"
OUTPUT_DIR = "data/prepared"
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
SEED = 42
SPLIT_RATIO = 0.8
SAMPLES = 10


class TextDataset(Dataset):
    def __init__(self, df):
        self.x = df["Review Text"][:SAMPLES].reset_index(drop=True)
        self.y = df["Polarity"][:SAMPLES].reset_index(drop=True)

    def __getitem__(self, index):
        return self.y[index], self.x[index]

    def __len__(self):
        return len(self.x)


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer.tokenize(str(text).lower())


def main():
    global SEED, SPLIT_RATIO, SAMPLES

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SEED = params["prepare"]["seed"]
    SPLIT_RATIO = params["prepare"]["split_ratio"]
    SAMPLES = params["prepare"]["samples"]

    torch.manual_seed(SEED)

    df = pd.read_csv(RAW_DATA_PATH, encoding="cp1256")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = TextDataset(df)
    train_size = int(SPLIT_RATIO * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1)
    vocab = build_vocab_from_iterator(
        yield_tokens(train_loader, tokenizer), specials=["<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    with open(os.path.join(OUTPUT_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(dict(vocab.get_stoi()), f)

    torch.save(list(train_set), os.path.join(OUTPUT_DIR, "train.pt"))
    torch.save(list(val_set), os.path.join(OUTPUT_DIR, "val.pt"))
    torch.save(list(test_set), os.path.join(OUTPUT_DIR, "test.pt"))


if __name__ == "__main__":
    main()
