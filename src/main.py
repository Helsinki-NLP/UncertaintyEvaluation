import sys
import time
import logging 
import click

import numpy as np
import torch

from datasets import load_dataset
from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TextClassificationPipeline,
        )

from metrics import *

# --- Auxilary functions ---
def run_metrics(
        predictions: np.array, 
        targets: np.array, 
        metrics: list):
    ''' call metric functions from src/metrics.py'''
    logging.info(f'Running metrics {metrics}')



# --- Click functions ---
@click.group()
def cli():
    pass

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


cli.add_command(eval_hf)
cli.add_command(eval_jax)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cli()
