from __future__ import annotations

import pickle
from dotenv import load_dotenv

load_dotenv()
import bentoml
import numpy as np
import onnxruntime as ort
import yaml
from pydantic import BaseModel, Field


with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=4, max_length=2000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


@bentoml.service(name="arabic_sentiment_service", resources={"cpu": "2"}, http={"port": 8000})
class ArabicSentimentModel:
    def __init__(
        self,
        tokenizer_model_name="aubmindlab/bert-base-arabertv02",
        vocab_file_path="notebooks/vocab.pkl",
        model_file_path="notebooks/model.onnx",
    ):
        from transformers import AutoTokenizer

        self.session = ort.InferenceSession(model_file_path)
        with open(vocab_file_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.classes = {0: "Negative", 1: "Positive"}
        self.version = params["version"]
        self.max_token_val = 7968

    def text_pipeline(self, text: str) -> list[int]:
        return [
            # self.vocab.get(token, 0)
            0
            if self.vocab.get(token, 0) > self.max_token_val
            else self.vocab.get(token, 0)
            for token in self.tokenizer.tokenize(str(text).lower())
        ]

    @bentoml.api(
        batchable=True,
    )
    def predict(self, requests: list[PredictRequest]) -> list[PredictResponse]:
        text_list = []
        lengths = []

        for req in requests:
            tokens = self.text_pipeline(req.text)
            if not tokens:
                tokens = [0]
            text_list.extend(tokens)
            lengths.append(len(tokens))

        onnx_text = np.array(text_list, dtype=np.int64)
        offsets = [0] + lengths[:-1]
        onnx_offsets = np.array(offsets, dtype=np.int64).cumsum()

        inputs = {"text": onnx_text, "offsets": onnx_offsets}
        logits = self.session.run(None, inputs)[0]

        responses = []
        preds = np.argmax(logits, axis=1)

        for idx, pred_idx in enumerate(preds):
            label = self.classes[pred_idx]
            prob = float(logits[idx][pred_idx])

            responses.append(
                PredictResponse(
                    label=label, confidence=round(prob, 5), model_version=self.version
                )
            )

        return responses
