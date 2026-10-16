import json
import os
import pickle
import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml


with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

PREPARED_DIR = "data/prepared"
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.01
EMBED_DIM = 64
SEED = 42


class TextClassificationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 20000,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_class: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, mode="mean", sparse=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class),
        )
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, text, offsets):
        x = self.embedding(text, offsets)
        x = self.norm(x)
        return self.classifier(x)


def get_collate_fn(tokenizer, vocab_dict):
    def text_pipeline(x):
        return [
            vocab_dict.get(token, vocab_dict.get("<unk>", 0))
            for token in tokenizer.tokenize(str(x).lower())
        ]

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(int(_label))
            tokens = text_pipeline(_text)
            if not tokens:
                tokens = [0]
            processed_text = torch.tensor(tokens, dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))

        return (
            torch.tensor(label_list, dtype=torch.int64),
            torch.cat(text_list),
            torch.tensor(offsets[:-1]).cumsum(dim=0),
        )

    return collate_batch


def evaluate(model, dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    cum_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for label, text, offsets in dataloader:
            label, text, offsets = label.to(device), text.to(device), offsets.to(device)
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            cum_loss += loss.item()
    return total_acc / total_count if total_count > 0 else 0.0, cum_loss


def main():
    global EPOCHS, LR, SEED, BATCH_SIZE, EMBED_DIM, SPLIT_RATIO, SAMPLES

    EPOCHS = params["train"]["epochs"]
    LR = params["train"]["lr"]
    SEED = params["train"]["seed"]
    BATCH_SIZE = params["train"]["batch_size"]
    EMBED_DIM = params["train"]["embed_dim"]
    SPLIT_RATIO = params["prepare"]["split_ratio"]
    SAMPLES = params["samples"]
    version = params["version"]

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    with open(os.path.join(PREPARED_DIR, "vocab.pkl"), "rb") as f:
        vocab_dict = pickle.load(f)

    train_data = torch.load(os.path.join(PREPARED_DIR, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(PREPARED_DIR, "val.pt"), weights_only=False)
    test_data = torch.load(os.path.join(PREPARED_DIR, "test.pt"), weights_only=False)

    collate_fn = get_collate_fn(tokenizer, vocab_dict)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    num_class = len({label for label, _ in train_data})
    model = TextClassificationModel(len(vocab_dict), EMBED_DIM, num_class).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("Model Training 1")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "lr": LR,
                "optimizer": "SGD",
                "epochs": EPOCHS,
                "emsize": EMBED_DIM,
                "seed": SEED,
                "split_ratio": SPLIT_RATIO,
                "samples": SAMPLES,
                "version": version,
            }
        )

        best_val_acc = -1
        for epoch in range(1, EPOCHS + 1):
            model.train()
            cum_loss = 0.0
            for label, text, offsets in train_loader:
                label, text, offsets = (
                    label.to(device),
                    text.to(device),
                    offsets.to(device),
                )
                optimizer.zero_grad()
                predicted = model(text, offsets)
                loss = criterion(predicted, label)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                cum_loss += loss.item()

            scheduler.step()
            val_acc, val_loss = evaluate(model, val_loader, device)

            mlflow.log_metric("train_loss", cum_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print(val_acc, best_val_acc, cum_loss)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    serialization_format="pickle",
                )

        test_acc, _ = evaluate(model, test_loader, device)
        mlflow.log_metric("test_accuracy", test_acc)

        baseline_samples = train_data[:1000]
        baseline_loader = DataLoader(
            baseline_samples,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )
        text_lengths = [len(str(text)) for _, text in baseline_samples]
        confidence_scores = []
        model.eval()
        with torch.no_grad():
            for _, text, offsets in baseline_loader:
                logits = model(text.to(device), offsets.to(device))
                confidence_scores.extend(
                    torch.softmax(logits, dim=1).max(dim=1).values.cpu().tolist()
                )

        with open("data/training_baseline.json", "w") as f:
            json.dump(
                {
                    "text_lengths": text_lengths,
                    "confidence_scores": confidence_scores,
                },
                f,
            )


if __name__ == "__main__":
    main()
