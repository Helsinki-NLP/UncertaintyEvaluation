import numpy as np
import pandas as pd
from datasets import load_dataset


mnli_train, mnli_dev_m, mnli_dev_mm = load_dataset('multi_nli', split=['train', 'validation_matched', 'validation_mismatched'], cache_dir='cache')
chaosnli_test = load_dataset("metaeval/chaos-mnli-ambiguity", split=['train'], cache_dir='cache')

mnli_train_df = pd.DataFrame(mnli_train)
mnli_dev_m_df = pd.DataFrame(mnli_dev_m)
mnli_dev_mm_df = pd.DataFrame(mnli_dev_mm)
chaosnli_test_df = pd.DataFrame(chaosnli_test)

print(chaosnli_test_df)

print('chaosnli features', chaosnli_test_df.columns)


print('MNLI Train vs ChaosMNLI:')
print(pd.Series(np.intersect1d(mnli_train_df['pairID'], chaosnli_test_df['uid'])))
print('\n')

print('MNLI Dev-m vs ChaosMNLI:')
print(pd.Series(np.intersect1d(mnli_dev_m_df['pairID'], chaosnli_test_df['uid'])))
print('\n')

print('MNLI Dev-mm vs ChaosMNLI:')
print(pd.Series(np.intersect1d(mnli_dev_mm_df['pairID'], chaosnli_test_df['uid'])))
