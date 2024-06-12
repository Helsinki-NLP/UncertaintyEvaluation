# UncertaintyEvaluation

unceval hf-text-classifier /scratch/project_2007780/hande/UncertaintyProject2024/models/swag/bert/snli/checkpoint-10000/ datasets/snli_annotations/ --num-predictions 20 --metric metrics/unceval_crosse
ntropy --metric metrics/unceval_accuracy
{'crossentropy': 0.7934141780321474, 'accuracy': 0.8226, 'total_time_in_seconds': 421.56179191730917, 'samples_per_second': 23.72131486233348, 'latency_in_seconds': 0.04215617919173092}


## BASELINE RESULTS - NODALIDA PAPER
| Train Dataset | Test Dataset | Method | Acc (%) | Cross-Entropy (lower better) |
| SNLI | SNLI (5 annotations per test) | vanilla BERT | 90.80 | 0.83 |
| SNLI | SNLI (5 annotations per test) | SWA | 91.47 | 0.75 |
| SNLI | SNLI (5 annotations per test) | SWAG | 91.59 | 0.69 |


## HF SWAG\_TRANSFORMERS + EVALUATION PACKAGE RESULTS
| SNLI | SNLI (5 annotations per test) | SWAG | 0.82 | 0.79 |

