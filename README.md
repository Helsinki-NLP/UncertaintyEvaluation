# UncertaintyEvaluation

### Text Classification Task

#### Datasets
SNLI: 5 annotations for validation and test splits
chaos-MNLI: 100 annotations, test only
chaos-SNLI: 100 annotations, test only

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
