import pickle
from abc import ABC, abstractmethod

import numpy as np
import onnx
import onnxruntime as ort
import scipy
import yaml
from transformers import AutoTokenizer

from config import settings

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


class ModelBase(ABC):
    @abstractmethod
    def predict(self, X: list[float]) -> list[float]: ...


class ArabicSentimentModel(ModelBase):
    def __init__(
        self,
        tokenizer_model_name=settings.MODEL_TOKENIZER,
        vocab_file_path=str(settings.MODEL_VOCAB_PATH),
        model_file_path=str(settings.MODEL_ONNX_PATH),
    ):
        super().__init__()
        self.session = ort.InferenceSession(model_file_path)
        with open(vocab_file_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.classes = {0: "Negative", 1: "Positive"}
        self.version = params["version"]
        self.max_token_val = self._get_vocab_limit(model_file_path)

    def _get_vocab_limit(self, model_file_path):
        model = onnx.load(model_file_path)
        for init in model.graph.initializer:
            if "embedding.weight" == init.name.lower() and len(init.dims) >= 2:
                return init.dims[0]
        return 14973

    def text_pipeline(self, text: str) -> list[int]:
        return [
            # self.vocab.get(token, 0)
            0
            if self.vocab.get(token, 0) > self.max_token_val
            else self.vocab.get(token, 0)
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
        return (
            self.classes[pred],
            scipy.special.softmax(logits, axis=1)[0][pred],
            self.version,
        )
