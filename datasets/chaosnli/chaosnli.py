# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""ChaosNLI dataset - SNLI and MNLI-matched portions

Usage:

    from datasets import load_dataset
    dataset = load_dataset('PATH/TO/MY/SCRIPT.py')
"""


import collections
import json
import os

import datasets


_CITATION = """\
@inproceedings{ynie2020chaosnli,
	Author = {Yixin Nie and Xiang Zhou and Mohit Bansal},
	Booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
	Publisher = {Association for Computational Linguistics},
	Title = {What Can We Learn from Collective Human Opinions on Natural Language Inference Data?},
	Year = {2020}
}
"""

_DESCRIPTION = """\
SNLI and MNLI-m portions of the ChaosSNLI dataset, which is a dataset with 100 annotations per example (a total of 4,645 * 100 annotations) for some existing data points in the development set of SNLI, MNLI, and Abductive NLI.
"""

# Direct link for Dropbox
# Eventually we wont need this, when the dataset is uplaoded to HF with their own format (arrow)
_DATA_URL = "https://www.dropbox.com/scl/fi/06nj75xq9djj5g72tzbmp/chaosNLI_v1.0.zip?rlkey=ihbteiy7nmgn79xplz9oicmj5&dl=1"


class Chaosnli(datasets.GeneratorBasedBuilder):
    """ChaosNLI dataset"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text import of ChaosNLI-SNLI portion",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "uid": datasets.Value("string"),
                    "label_counter": {
                        "e": datasets.Value("int32"),
                        "n": datasets.Value("int32"),
                        "c": datasets.Value("int32")
                    },
                    "majority_label": datasets.Value("string"),
                    "label_dist":  datasets.features.Sequence(datasets.Value("float")),
                    "label_count":  datasets.features.Sequence(datasets.Value("int32")),
                    "entropy": datasets.Value("float"),
                    "example": {
                        "uid": datasets.Value("string"),
                        "premise": datasets.Value("string"),
                        "hypothesis": datasets.Value("string"),
                        "source": datasets.Value("string")
                    },
                    "old_label": datasets.Value("string"),
                    "old_labels": datasets.features.Sequence(feature=datasets.features.ClassLabel(names=["entailment", "neutral", "contradiction"]))
                }
            ),
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://github.com/easonnie/ChaosNLI",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_manager.download_config.force_download = True
        dl_dir = dl_manager.download_and_extract(_DATA_URL)
        data_dir = os.path.join(dl_dir, "chaosNLI_v1.0")
        return [
            datasets.SplitGenerator(
                name="snli", gen_kwargs={"filepath": os.path.join(data_dir, "chaosNLI_snli.jsonl")}
            ),
            datasets.SplitGenerator(
                name="mnli_m", gen_kwargs={"filepath": os.path.join(data_dir, "chaosNLI_mnli_m.jsonl")}
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                yield idx, {
                    "premise": row["example"]["premise"],
                    "hypothesis": row["example"]["hypothesis"],
                    "label": row["majority_label"],
                    "label_counter": row["label_counter"],
                    "label_count": row["label_count"],
                    "label_dist": row["label_dist"],
                    "entropy": row["entropy"],
                    "old_label": row["old_label"],
                    "old_labels": row["old_labels"]
                }
