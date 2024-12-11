"""Command-line tools"""

import importlib
import logging

import click
import evaluate
import datasets

from .utils import import_object


logger = logging.getLogger(__name__)


REFERENCE_COLUMN = "label_dist"


# --- Click functions ---
@click.group()
@click.option('-v', 'verbosity', count=True, help='Increase logging verbosity')
def cli(verbosity: int = 0):
    level = logging.WARNING - 10 * verbosity
    logging.basicConfig(level=level)


@cli.command('hf-text-classifier')
@click.argument('model_path', required=True, type=str)
@click.argument('dataset_path', required=True, type=str)
@click.option('--dataset-split', type=str, default='test', help='dataset split to use')
@click.option('--dataset-limit', type=int, default=None, help='limit to N first data samples')
@click.option('--dataset-column', type=str, default=REFERENCE_COLUMN,
              help='dataset column for reference label distribution')
@click.option('--dataset-collapse', 'dataset_collapse', nargs=2, type=str, default=(None, None),
              help=('collapse row-per-annotation dataset for label distribution; '
                    'the arguments are columns for sample identifier and label'))
@click.option('--metric', 'metrics', multiple=True, type=str, help='list of metrics to run')
@click.option('--batch-size', type=int, default=10, help='batch size')
@click.option('--num-predictions', type=int, default=10,
              help='requested number of predictions per sample (may be resticted by the model)')
@click.option('--register-custom', nargs=3, type=str, multiple=True,
              help=('Register a custom class for auto loaders. The arguments are: '
                    '<name> <config class path> <model class path>. The paths should start '
                    'the module and end with the class name. Example: '
                    '--register-custom my_bert mymodule.MyBertConfig mymodule.MyBertModel'))
def hf_text_classification(model_path, dataset_path, metrics, dataset_split, dataset_limit,
                           dataset_column, dataset_collapse, batch_size, num_predictions,
                           register_custom):
    """Run text classification task"""
    # Dynamic imports
    transformers = importlib.import_module("transformers")
    text_classification = importlib.import_module("..text_classification", package=__name__)

    for name, cfg_class_path, model_class_path in register_custom:
        cfg_class = import_object(cfg_class_path)
        model_class = import_object(model_class_path)
        transformers.AutoConfig.register(name, cfg_class)
        transformers.AutoModelForSequenceClassification.register(cfg_class, model_class)

    dataset = datasets.load_dataset(dataset_path, split=dataset_split)
    logger.info("Original dataset: %s", dataset)

    dataset, input_column, second_input_column = text_classification.prepare_dataset(
        dataset, dataset_limit, dataset_column, dataset_collapse[0], dataset_collapse[1])
    logger.info("Dataset: %s", dataset)

    if dataset_column not in dataset.features:
        raise ValueError(f"Dataset {dataset_path} does not include '{dataset_column}' feature")

    num_labels = len(dataset[dataset_column][0])
    logger.info("Number of labels detected: %s", num_labels)

    metrics = [evaluate.load(metric) for metric in metrics]
    if metrics:
        logger.info("Metrics: %s", [metric.name for metric in metrics])
    else:
        logger.warning("No metrics set!")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels)

    pipe = text_classification.TextClassificationUncertaintyPipeline(
        model=model, tokenizer=tokenizer, task='text-classification',
        batch_size=batch_size, num_predictions=num_predictions)

    task_evaluator = text_classification.TextClassificationUncertaintyEvaluator()

    logger.info("Computing and evaluating predictions")
    eval_results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=dataset,
        input_column=input_column,
        second_input_column=second_input_column,
        label_column=dataset_column,
        metric=evaluate.combine(metrics),
    )
    print(eval_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cli()
