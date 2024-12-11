# UncertaintyEvaluation

## Instructions

This package provides the `unceval` command that has subcommands. The
subcommand implemented at this point is `hf-text-classifier` for
evaluating models build with the Hugging Face (HF) transformer models
for text classification. In addition to the command line interface and
related code, the package contains some datasets and metrics under the
respective directories.

Example for evaluating an NLI model with the included SNLI dataset
using accuracy and cross-entropy:
```
unceval hf-text-classifier \
/path/to/nli_model \
datasets/snli_annotations \
--metric metrics/unceval_accuracy \
--metric metrics/unceval_crossentropy
```

The models, datasets, and evaluation metrics can be loaded directly
from HF's respective hubs (`models`, `datasets`, `evaluate`), but note
that the selected combination has to be compatible. For example, the
metrics in this repository require that the dataset has a column with
human label distributions (instead of a single label).

The `--register-custom` option can be used to add support for custom
model types outside the HF `transformers` library. For example, the SWAG
transformer extension for BERT from [swag_transformers](https://github.com/Helsinki-NLP/swag_transformers/)
can be used like this:
```
unceval hf-text-classifier \
--register-custom swag_bert swag_transformers.swag_bert.SwagBertConfig swag_transformers.swag_bert.SwagBertForSequenceClassification \
[...]
```

## Results

### Text Classification Task

#### Datasets
SNLI: 5 annotations for validation and test splits chaos-MNLI: 100
annotations, test only chaos-SNLI: 100 annotations, test only

##### Baseline Results I (Nodalida Paper)
| Train Dataset | Test Dataset | Method | Acc (%) | Cross-Entropy (lower better) |
| ------------- | ------------ | ------ | ------- | ---------------------------- |
| SNLI | SNLI | vanilla BERT | 90.80 | 0.83 |
| SNLI | SNLI | SWA | 91.47 | 0.75 |
| SNLI | SNLI | SWAG | 91.59 | 0.69 |
| MNLI | MNLI-m | vanilla BERT | 86.53 | 0.87 |
| MNLI | MNLI-m | SWA | 87.60 | 0.80 |
| MNLI | MNLI-m | SWAG | 87.76 | 0.73 |
| MNLI | MNLI-mm | vanilla BERT | 86.31 | 0.84 |
| MNLI | MNLI-mm | SWA | 87.34 | 0.77 |
| MNLI | MNLI-mm | SWAG | 87.51 | 0.69 |
| SNLI | MNLI-m | vanilla BERT | 77.31 | 1.13 |
| SNLI | MNLI-m | SWA | 79.67 | 0.90 |
| SNLI | MNLI-m | SWAG | 79.33 | 0.80 |
| SNLI | MNLI-mm | vanilla BERT | 77.40 | 1.12 |
| SNLI | MNLI-mm | SWA | 79.44 | 0.88 |
| SNLI | MNLI-mm | SWAG | 79.24 | 0.79 |

##### Baseline Results II (Elaine's Results from Notion.so)
| Train Dataset | Test Dataset | Method | Acc (%) | Cross-Entropy (lower better) |
| ------------- | ------------ | ------ | ------- | ---------------------------- |
| MNLI-half | chaos-MNLI | vanilla BERT | 50.28 | 1.02 |
| MNLI-half | MNLI | vanilla BERT | 76.88 | 0.83 |

##### Baseline Results III (Third-Party Models)
| Train Dataset | Test Dataset | Method | Acc (%) | Cross-Entropy (lower better) |
| ------------- | ------------ | ------ | ------- | ---------------------------- |
| SNLI | SNLI | textattack/bert-base-uncased-snli | 90.48 | - | 

##### Evaluation package results
| Train Dataset | Test Dataset | Method | Acc (%) | Cross-Entropy (lower better) |
| ------------- | ------------ | ------ | ------- | ---------------------------- |
| SNLI | SNLI\_soft\_annotated | Our HF SWAG | 82.26 | 0.79 |
| SNLI | SNLI\_soft\_annotated | textattack/bert-base-uncased-snli | 4.46 | 6.04 | 
| SNLI | chaos-MNLI | Our HF SWAG | 46.15 | 1.52 |
