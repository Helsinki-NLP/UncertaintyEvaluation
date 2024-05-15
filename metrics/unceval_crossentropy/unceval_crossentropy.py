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
"""Cross-entropy metric for label distributions."""

import datasets
import evaluate
import logging
import numpy as np


_DESCRIPTION = """
Cross-entropy for sampled prediction probabilities over distribution of labels
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of lists of lists of `float`): Sampled predicted label probabilities per each data point.
    references (`list` of lists of `float`): Ground truth label probabilities per each data point.
    normalize (`boolean`): If true, return the mean loss per sample. Otherwise, return the sum of the per-sample losses.

Returns:
    crossentropy (`float`): Cross-entropy score. Minimum possible value is 0, no maximum possible value. A higher score means higher cross-entropy.
"""


_CITATION = """
"""


logger = logging.getLogger(__name__)


def entropy_score(ref_probs, pred_probs, normalize=True):
    total, hsum = 0, 0.0
    for ref, pred in zip(ref_probs, pred_probs):
        mean_pred = np.mean(pred, axis=0)
        value = -np.sum(np.array(ref) * np.log(mean_pred))
        hsum += value
        total += 1
    if normalize:
        hsum /= total
    return hsum


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CrossEntropy(evaluate.Metric):
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
            "crossentropy": float(
                entropy_score(references, predictions, normalize=normalize)
            )
        }
