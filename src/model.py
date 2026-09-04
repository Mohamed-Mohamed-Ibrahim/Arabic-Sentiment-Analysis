from abc import ABC, abstractmethod


class ModelBase(ABC):
    @abstractmethod
    def predict(self, X: list[float]) -> list[float]: ...


class ArabicSentimentModel(ModelBase):
    def __init__(self):
        super().__init__()
    
    def predict(X):
        pass