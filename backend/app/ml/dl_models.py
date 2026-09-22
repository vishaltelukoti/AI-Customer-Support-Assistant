"""Small Day 4 neural text models for category-only comparison."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .preprocessing import SEED


@dataclass(frozen=True)
class DLConfig:
    vocabulary_size: int = 5000
    sequence_length: int = 60
    embedding_dim: int = 32
    hidden_dim: int = 32
    dense_max_iter: int = 200


class SimpleTokenizer:
    def __init__(self, vocabulary_size: int):
        self.vocabulary_size = vocabulary_size
        self.word_index_: dict[str, int] = {}

    def fit(self, texts: list[str]) -> "SimpleTokenizer":
        counts: dict[str, int] = {}
        for text in texts:
            for token in text.split():
                counts[token] = counts.get(token, 0) + 1
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        self.word_index_ = {token: index + 1 for index, token in enumerate(ordered[: self.vocabulary_size - 1])}
        return self

    def transform(self, texts: list[str], sequence_length: int) -> np.ndarray:
        sequences = np.zeros((len(texts), sequence_length), dtype=np.int32)
        for row, text in enumerate(texts):
            ids = [self.word_index_.get(token, 0) for token in text.split()[:sequence_length]]
            if ids:
                sequences[row, : len(ids)] = ids
        return sequences


class NeuralTextClassifier:
    def __init__(self, name: str, config: DLConfig = DLConfig(), seed: int = SEED):
        self.name = name
        self.config = config
        self.seed = seed
        self.tokenizer = SimpleTokenizer(config.vocabulary_size)
        self.classes_: np.ndarray | None = None
        self.classifier = LogisticRegression(max_iter=config.dense_max_iter, random_state=seed)
        rng = np.random.default_rng(seed)
        self.embedding = rng.normal(0, 0.05, size=(config.vocabulary_size, config.embedding_dim)).astype(np.float32)
        self.embedding[0] = 0
        self.weights = self._weights(rng)

    def _weights(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        c = self.config
        if self.name == "cnn":
            return {"kernel": rng.normal(0, 0.08, size=(3, c.embedding_dim, c.hidden_dim)).astype(np.float32)}
        if self.name == "rnn":
            return {
                "wx": rng.normal(0, 0.08, size=(c.embedding_dim, c.hidden_dim)).astype(np.float32),
                "wh": rng.normal(0, 0.08, size=(c.hidden_dim, c.hidden_dim)).astype(np.float32),
            }
        if self.name == "lstm":
            return {
                "wx": rng.normal(0, 0.08, size=(c.embedding_dim, 4 * c.hidden_dim)).astype(np.float32),
                "wh": rng.normal(0, 0.08, size=(c.hidden_dim, 4 * c.hidden_dim)).astype(np.float32),
            }
        raise ValueError(f"Unsupported model name: {self.name}")

    def _features(self, texts: list[str]) -> np.ndarray:
        sequences = self.tokenizer.transform(texts, self.config.sequence_length)
        embedded = self.embedding[sequences]
        return FEATURE_EXTRACTORS[self.name](embedded, self.weights)

    def fit(self, texts: list[str], labels: list[str]) -> "NeuralTextClassifier":
        self.tokenizer.fit(texts)
        self.classes_ = np.asarray(sorted(set(labels)))
        self.classifier.fit(self._features(texts), labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.classifier.predict(self._features(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.classifier.predict_proba(self._features(texts))


def _cnn_features(embedded: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    kernel = weights["kernel"]
    padded = np.pad(embedded, ((0, 0), (1, 1), (0, 0)))
    windows = [padded[:, index:index + 3, :] for index in range(embedded.shape[1])]
    activations = np.stack([np.tensordot(window, kernel, axes=([1, 2], [0, 1])) for window in windows], axis=1)
    return np.maximum(activations, 0).max(axis=1)


def _rnn_features(embedded: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    hidden = np.zeros((embedded.shape[0], weights["wh"].shape[0]), dtype=np.float32)
    for step in range(embedded.shape[1]):
        hidden = np.tanh(embedded[:, step, :] @ weights["wx"] + hidden @ weights["wh"])
    return hidden


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def _lstm_features(embedded: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    hidden = np.zeros((embedded.shape[0], weights["wh"].shape[0]), dtype=np.float32)
    cell = np.zeros_like(hidden)
    for step in range(embedded.shape[1]):
        gates = embedded[:, step, :] @ weights["wx"] + hidden @ weights["wh"]
        i, f, g, o = np.split(gates, 4, axis=1)
        cell = _sigmoid(f) * cell + _sigmoid(i) * np.tanh(g)
        hidden = _sigmoid(o) * np.tanh(cell)
    return hidden


FEATURE_EXTRACTORS: dict[str, Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray]] = {
    "cnn": _cnn_features,
    "rnn": _rnn_features,
    "lstm": _lstm_features,
}


def build_dl_model(model_name: str, config: DLConfig = DLConfig(), seed: int = SEED) -> NeuralTextClassifier:
    return NeuralTextClassifier(model_name, config, seed)
