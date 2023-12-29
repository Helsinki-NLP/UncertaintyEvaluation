"""Text classification"""

import inspect
import logging

import evaluate
import torch
import transformers
from transformers.pipelines.text_classification import softmax


logger = logging.getLogger(__name__)


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
            return self.model.get_logits(model_inputs["input_ids"], num_predictions=num_predictions)
        outputs = self.model(**model_inputs)
        logits = outputs['logits'].tile((num_predictions, 1, 1))
        logits = torch.permute(logits, (1, 0, 2))
        logger.debug("Transforming logits from %s to %s", outputs['logits'].shape, logits.shape)
        return logits

    def postprocess(self, model_outputs):
        logger.debug("Model outputs: %s", model_outputs)
        outputs = model_outputs.numpy()
        scores = softmax(outputs)
        logger.debug("Scores: %s", scores)
        return scores


class TextClassificationUncertaintyEvaluator(evaluate.TextClassificationEvaluator):

    def predictions_processor(self, predictions, label_mapping):
        logger.debug("Predictions: %s", len(predictions))
        logger.debug("Predictions shape: %s", predictions[0].shape)
        return {"predictions": predictions}
