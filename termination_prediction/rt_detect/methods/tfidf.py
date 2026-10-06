"""TF-IDF + Logistic Regression / Ridge supervised baseline."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from rt_detect.data import format_context
from rt_detect.methods._common import _make_pred


def run_tfidf(train_examples, test_examples, label_mode,
              include_victim=True, include_initial_contact=True):
    def texts(exs):
        return [format_context(ex, include_victim, include_initial_contact)
                for ex in exs]

    label_key    = "label_binary" if label_mode == "binary" else "label_regression"
    train_labels = [ex[label_key] for ex in train_examples]

    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                            sublinear_tf=True)
    if label_mode == "binary":
        pipe = Pipeline([("tfidf", tfidf),
                         ("clf",   LogisticRegression(max_iter=1000, C=1.0))])
        pipe.fit(texts(train_examples), train_labels)
        scores = pipe.predict_proba(texts(test_examples))[:, 1].tolist()
    else:
        pipe = Pipeline([("tfidf", tfidf),
                         ("reg",   Ridge(alpha=1.0))])
        pipe.fit(texts(train_examples), train_labels)
        scores = np.clip(pipe.predict(texts(test_examples)), 0.0, 1.0).tolist()

    return [_make_pred(ex, float(s)) for ex, s in zip(test_examples, scores)], pipe


def save_tfidf(pipe, out_dir):
    path = Path(out_dir) / "tfidf_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(pipe, f)
    return path
