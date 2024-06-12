# UncertaintyEvaluation

### Text Classification Task

#### SNLI dataset
5 annotations for validation and test splits

##### Baseline Results from Nodalida Paper
| Train Dataset | Test Dataset | Method | Acc (%) | Cross-Entropy (lower better) |
| ------------- | ------------ | ------ | ------- | ---------------------------- |
| SNLI | SNLI (5 annotations per test) | vanilla BERT | 90.80 | 0.83 |
| SNLI | SNLI (5 annotations per test) | SWA | 91.47 | 0.75 |
| SNLI | SNLI (5 annotations per test) | SWAG | 91.59 | 0.69 |

##### Evaluation package results (Trained by HF SWAG package)
| Train Dataset | Test Dataset | Method | Acc (%) | Cross-Entropy (lower better) |
| ------------- | ------------ | ------ | ------- | ---------------------------- |
| SNLI | SNLI (5 annotations per test) | SWAG | 82.26 | 0.79 |

