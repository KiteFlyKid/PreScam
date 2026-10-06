"""Position-only baseline: uses only round_index as feature.

Sanity-check baseline that reveals how much predictability comes from
"the further along in the conversation, the higher the risk",
independent of any text content.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rt_detect.methods._common import _make_pred


def run_position_only(train_examples, test_examples, label_mode):
    label_key = "label_binary" if label_mode == "binary" else "label_regression"

    X_train = np.array([[ex["round_index"]] for ex in train_examples], dtype="float32")
    X_test  = np.array([[ex["round_index"]] for ex in test_examples],  dtype="float32")
    y_train = [ex[label_key] for ex in train_examples]

    if label_mode == "binary":
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf",    LogisticRegression(max_iter=1000))])
        pipe.fit(X_train, y_train)
        scores = pipe.predict_proba(X_test)[:, 1].tolist()
    else:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("reg",    Ridge())])
        pipe.fit(X_train, y_train)
        scores = np.clip(pipe.predict(X_test), 0.0, 1.0).tolist()

    return [_make_pred(ex, float(s)) for ex, s in zip(test_examples, scores)]
