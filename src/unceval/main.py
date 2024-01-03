"""Command-line tools"""

import importlib
import logging

import click
import evaluate
import datasets


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
@click.option('--metric', 'metrics', multiple=True, type=str, help='list of metrics to run')
@click.option('--batch-size', type=int, default=10, help='batch size')
@click.option('--num-predictions', type=int, default=10,
              help='requested number of predictions per sample (may be resticted by the model)')
def hf_text_classification(model_path, dataset_path, metrics, dataset_split, dataset_limit,
                           dataset_column, batch_size, num_predictions):
    # Dynamic imports
    transformers = importlib.import_module("transformers")
    text_classification = importlib.import_module("..text_classification", package=__name__)

    dataset = datasets.load_dataset(dataset_path, split=dataset_split)
    if dataset_limit:
        dataset = dataset.select(range(dataset_limit))
    logger.info(dataset)
    if dataset_column not in dataset.features:
        raise ValueError(f"Dataset {dataset_path} does not include '{dataset_column}' feature")
    if "premise" in dataset.features and "hypothesis" in dataset.features:
        logger.info("NLI task detected")
        input_column = "premise"
        second_input_column = "hypothesis"
    elif "text" in dataset.features:
        logger.info("Assuming text classification task with single input in 'text'")
        input_column = "text"
        second_input_column = None
    else:
        raise ValueError(f"Could not determine task from the dataset features {dataset.features}")

    metrics = [evaluate.load(metric) for metric in metrics]
    logger.info("Metrics: %s", [metric.name for metric in metrics])

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

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
