import pickle
from abc import ABC, abstractmethod

import numpy as np
import onnxruntime as ort
import yaml
from transformers import AutoTokenizer

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


class ModelBase(ABC):
    @abstractmethod
    def predict(self, X: list[float]) -> list[float]: ...


class ArabicSentimentModel(ModelBase):
    def __init__(
        self,
        tokenizer_model_name="aubmindlab/bert-base-arabertv02",
        vocab_file_path="notebooks/vocab.pkl",
        model_file_path="notebooks/model.onnx",
    ):
        super().__init__()
        self.session = ort.InferenceSession(model_file_path)
        with open(vocab_file_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.classes = {0: "Negative", 1: "Positive"}
        self.version = params["version"]

    def text_pipeline(self, text):
        return [
            self.vocab.get(token, 0)
            for token in self.tokenizer.tokenize(str(text).lower())
        ]

    def predict(self, X):
        text_list, offsets = [], [0]
        for text in X:
            tokens = self.text_pipeline(text)
            if not tokens:
                tokens = [0]
            text_list.extend(tokens)
            offsets.append(len(tokens))

        onnx_text = np.array(text_list, dtype=np.int64)
        onnx_offsets = np.array(offsets[:-1], dtype=np.int64).cumsum()

        inputs = {"text": onnx_text, "offsets": onnx_offsets}
        logits = self.session.run(None, inputs)[0]
        pred = np.argmax(logits, axis=1)[0]
        return self.classes[pred], logits[0][pred], self.version
