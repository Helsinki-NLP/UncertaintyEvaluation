import logging

import click
import evaluate
import datasets


logger = logging.getLogger(__name__)


# --- Click functions ---
@click.group()
@click.option('-v', 'verbosity', count=True, help='Increase logging verbosity')
def cli(verbosity: int = 0):
    level = logging.WARNING - 10 * verbosity
    logging.basicConfig(level=level)


@cli.command('hf-text-classifier')
@click.argument('model', required=True, type=str)
@click.argument('dataset', required=True, type=str)
@click.option('--dataset-split', type=str, default='test', help='dataset split to use')
@click.option('--dataset-limit', type=int, default=None, help='limit to N first data samples')
@click.option('--metric', 'metrics', multiple=True, type=str, help='list of metrics to run')
@click.option('--num-predictions', type=int, default=10, help='maximum number of predictions per sample')
def hf_text_classification(model, dataset, metrics, dataset_split, dataset_limit, num_predictions):
    # Dynamic imports
    import transformers
    from .text_classification import TextClassificationUncertaintyPipeline, \
        TextClassificationUncertaintyEvaluator

    dataset = datasets.load_dataset(dataset, split=dataset_split)
    if dataset_limit:
        dataset = dataset.select(range(dataset_limit))
    logger.info(dataset)

    metrics = [evaluate.load(metric) for metric in metrics]
    logger.info(metrics)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model)

    pipe = TextClassificationUncertaintyPipeline(
        model=model, tokenizer=tokenizer, task='text-classification',
        batch_size=5, num_predictions=num_predictions)

    task_evaluator = TextClassificationUncertaintyEvaluator()

    eval_results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=dataset,
        input_column="premise",
        second_input_column="hypothesis",
        label_mapping={"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
        label_column="label_dist",
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
