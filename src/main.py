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

from .metrics import *

# --- Auxilary functions ---
def run_metrics(
        predictions: np.array, 
        targets: np.array, 
        metrics: list[string]):
    ''' call metric functions from src/metrics.py'''
    logging.info(f'Running metrics {metrics}')



# --- Click functions ---
@click.group()
def cli():
    pass

@click.command()
@click.argument('model_type')
@click.option('--model', help='name of the HF model to evaluate')
@click.option('--dataset', help='name of the HF dataset to evaluate the model on')
@click.option('--metrics', help='list of metrics to run')
def eval_hf(): 
    assert model_type in ['classification', 'seq2seq'], 'Supported model types are [classification, seq2seq]'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    #Load model
    if model_type == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(model)
    elif model_type == 'seq2seq':
        raise NotImplementedError

    #Load dataset
    dataset = load_dataset(dataset)

    #Run the pipeline
    model.to(device)
    tokenizer.to(device)
    dataset.to(device)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)


    #Call evaluate function with model, dataset, list of metrics
    run_metrics(predictions, targets, metrics)

@click.command()
def eval_jax():
    click.echo('Evaluating JAX model')


cli.add_command(eval_hf)
cli.add_command(eval_jax)

if __name__ == "__main__":
    cli()
