"""Utility functions"""

import collections

import datasets
import tqdm


def collapse_dataset(original, input_column, second_input_column, id_column,
                     label_column, label_dist_column):
    """Collapse dataset with each annotation in a separate row

    In the output dataset, each id_column value will have one row,
    that includes the input columns and a distribution of labels in
    label_column in the new column label_dist_column.

    Returns the new dataset and a mapping of labels to the their index
    in label_dist.

    """
    labels = collections.Counter(original[label_column])
    num_labels = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(sorted(labels.keys()))}
    new = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in tqdm.tqdm(original):
        sample_id = row[id_column]
        label = row[label_column]
        if sample_id not in new:
            new[sample_id][id_column] = sample_id
            new[sample_id][label_dist_column] = [0] * num_labels
            for col in [input_column, second_input_column]:
                if col is None:
                    continue
                new[sample_id][col] = row[col]
        new[sample_id][label_dist_column][label_to_idx[label]] += 1
    return datasets.Dataset.from_generator(new.values), label_to_idx
