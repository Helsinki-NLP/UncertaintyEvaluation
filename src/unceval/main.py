"""Command-line tools"""

import importlib
import logging
import random

import click
import evaluate
import datasets
from transformers import TextClassificationPipeline

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
#@click.option("--kwarg", type=(str, float), multiple=True,
#              help=('model-specific parameters'))   
@click.option('--method', type=str, default='swag-diag', help='from {swa, swag, swag-block, swag-diag, swag-scale-1, base}')
@click.option('--seed', type=int, default=491)
def hf_text_classification(model_path, dataset_path, metrics, dataset_split, dataset_limit,
                           dataset_column, dataset_collapse, batch_size, num_predictions,
                           register_custom, method, seed):
    """Run text classification task"""
    logging.basicConfig(level=logging.INFO)
    
    # Set random seed
    random.seed(seed)

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

    #FIXME: Hande: I think this is relevant only for entropy measuring, so need to check if in metrics before this
    #if dataset_column not in dataset.features:
    #    raise ValueError(f"Dataset {dataset_path} does not include '{dataset_column}' feature")

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

    if method == 'swa':
        cov = False
        scale = 0
        block = False
    elif method == 'swag':
        cov = True
        scale = 0.5
        block = False
    elif method == 'swag-scale-1':
        cov = True
        scale = 1
        block = False
    elif method == 'swag-block':
        cov = True
        scale = 0.5
        block = True
    elif method == 'swag-diag':
        cov = False
        scale = 1.0
        block = False


    print(f'method: {method}, eval parameters: scale: {scale}, cov: {cov}, block: {block}, seed:{seed}')
    logger.info("Computing and evaluating predictions")
    
    if method == "base":
        pipe = TextClassificationPipeline(
            model=model, tokenizer=tokenizer, task='text-classification',
            batch_size=batch_size, 
            )

        task_evaluator = evaluator("text-classification")

    else:
        pipe = text_classification.TextClassificationUncertaintyPipeline(
            model=model, tokenizer=tokenizer, task='text-classification',
            batch_size=batch_size, 
            num_predictions=num_predictions,
            cov=cov,
            scale=scale,
            block=block
            )

        task_evaluator = text_classification.TextClassificationUncertaintyEvaluator()

    eval_results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=dataset,
        input_column=input_column,
        second_input_column=second_input_column,
        label_column=dataset_column,
        metric=evaluate.combine(metrics),
    )


    print(eval_results)


@click.command()
#@click.argument('task')
#@click.argument('model', required=True, type=str)
#@click.argument('dataset', required=True, type=str)
#@click.option('metrics', required=True, multiple=True)
@click.option('--task', 'task')
@click.option('--model', 'model_id')
@click.option('--dataset', 'dataset_id', help='name of the HF dataset to evaluate the model on')
@click.option('--metrics', multiple=True, help='list of metrics to run')
def eval_hf(task, model_id, dataset_id, metrics):
    logging.debug(f'task: {task}')
    logging.debug(f'model: {model_id}')
    logging.debug(f'dataset: {dataset_id}')
    logging.debug(f'metrics: {metrics}')

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TextClassificationPipeline,
    )

    # Should be used only for torch models?
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #Load dataset
    dataset = load_dataset(dataset_id, cache_dir='/home/hande/cache')

    #Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    #Load model
    if task == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    elif task == 'seq2seq':
        raise NotImplementedError

    #Run the pipeline
    model.to(device)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)


    #Call evaluate function with model, dataset, list of metrics
    run_metrics(predictions, targets, metrics)


@click.command()
def eval_jax():
    click.echo('Evaluating JAX model')


# Disabled
# cli.add_command(eval_hf)
# cli.add_command(eval_jax)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cli()
