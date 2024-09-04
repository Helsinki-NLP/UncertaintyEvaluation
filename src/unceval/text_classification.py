"""Text classification"""

import inspect
import logging

import evaluate
import torch
import transformers
from transformers.pipelines.text_classification import softmax

from unceval import utils


logger = logging.getLogger(__name__)


def prepare_dataset(dataset, dataset_limit, dataset_column, dataset_id_column=None,
                    dataset_original_label_column=None):
    """Prepare text classification dataset"""

    if "premise" in dataset.features and "hypothesis" in dataset.features:
        logger.info("NLI task detected")
        input_column = "premise"
        second_input_column = "hypothesis"
    elif "text" in dataset.features:
        logger.info("Assuming task with single input in 'text'")
        input_column = "text"
        second_input_column = None
    else:
        raise ValueError(f"Could not determine task from the dataset features {dataset.features}")

    if dataset_id_column and dataset_original_label_column:
        logger.info("Converting dataset to have row per unique value of %s", dataset_id_column)
        dataset, label_indices = utils.collapse_dataset(
            dataset, input_column, second_input_column, id_column=dataset_id_column,
            label_column=dataset_original_label_column, label_dist_column=dataset_column)
        logger.info("Label indices: %s", label_indices)

    if dataset_limit:
        dataset = dataset.select(range(dataset_limit))

    return dataset, input_column, second_input_column


class TextClassificationUncertaintyPipeline(transformers.pipelines.Pipeline):

    def _sanitize_parameters(self, num_predictions=None, **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        forward_params = {}
        postprocess_params = {}
        if num_predictions is not None:
            forward_params['num_predictions'] = num_predictions
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, input_, **tokenizer_kwargs):
        return transformers.pipelines.TextClassificationPipeline.preprocess(
            self, input_, **tokenizer_kwargs)

    def _forward(self, model_inputs, num_predictions=10):
        # `XXXForSequenceClassification` models should not use `use_cache=True` even if it's supported
        model_forward = self.model.forward if self.framework == "pt" else self.model.call
        if "use_cache" in inspect.signature(model_forward).parameters.keys():
            model_inputs["use_cache"] = False
        if hasattr(self.model, 'get_logits'):
            input_ids = model_inputs.pop('input_ids')
            logits = self.model.get_logits(input_ids, num_predictions=num_predictions, **model_inputs)
            if len(logits.shape) == 4:
                # [batch_size, preds, seq_len, output_size] -> [batch_size, preds, labels]
                logits = torch.squeeze(logits, 2)
            return logits
        outputs = self.model(**model_inputs)
        logits = outputs['logits']
        if len(logits.shape) == 2:
            # Expects original shape [batch_size, labels]
            logits = outputs['logits'].tile((num_predictions, 1, 1))  # [preds, batch_size, labels]
            logits = torch.permute(logits, (1, 0, 2))                 # [batch_size, preds, labels]
            logger.debug("Transforming logits from %s to %s", outputs['logits'].shape, logits.shape)
        return logits

    def postprocess(self, model_outputs):
        logger.debug("Model outputs: %s", model_outputs)
        outputs = model_outputs.numpy()
        scores = softmax(outputs)
        if len(scores.shape) == 3 and scores.shape[0] == 1:
            # Extra dimension returned if batch_size = 1
            scores = scores[0]
        logger.debug("Scores: %s", scores)
        return scores


class TextClassificationUncertaintyEvaluator(evaluate.TextClassificationEvaluator):

    def predictions_processor(self, predictions, label_mapping):
        logger.debug("Predictions: %s", len(predictions))
        logger.debug("Predictions shape: %s", predictions[0].shape)
        return {"predictions": predictions}
