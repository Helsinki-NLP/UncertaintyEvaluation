# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Accuracy metric for label distributions."""

import datasets
import evaluate
import logging
import numpy as np


_DESCRIPTION = """
Accuracy for sampled prediction probabilities given distribution of labels
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of lists of lists of `float`): Sampled predicted label probabilities per each data point.
    references (`list` of lists of `float`): Ground truth label probabilities per each data point.
    normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.

Returns:
    accuracy (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher score means higher accuracy.
"""


_CITATION = """
"""


logger = logging.getLogger(__name__)


def accuracy_score(ref_probs, pred_probs, normalize=True):
    total, hits = 0, 0
    for ref, pred in zip(ref_probs, pred_probs):
        ref_label = np.argmax(ref)
        mean_pred = np.mean(pred, axis=0)
        pred_label = np.argmax(mean_pred)
        if ref_label == pred_label:
            hits += 1
        total += 1
    if normalize:
        return hits / total
    return hits


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Accuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                    "references": datasets.Sequence(datasets.Value("float32")),
                }
            ),
            # reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )

    def _compute(self, predictions, references, normalize=True):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize)
            )
        }
