from typing import Any, Dict

from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score

from transformers_framework.architectures.modeling_outputs import TokenDetectionOutput
from transformers_framework.interfaces.adaptation import token_detection_adaptation
from transformers_framework.interfaces.logging import (
    LOSS,
    TOKEN_DETECTION_ACCURACY,
    TOKEN_DETECTION_F1,
    TOKEN_DETECTION_LOSS,
    TOKEN_DETECTION_PERPLEXITY,
)
from transformers_framework.interfaces.step import TokenDetectionStepOutput
from transformers_framework.metrics.perplexity import Perplexity
from transformers_framework.pipelines.pipeline.pipeline import Pipeline
from transformers_framework.processing.postprocessors import random_token_detection_processor
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_random_token_detection_arguments
from transformers_framework.utilities.distributions import expand_logits


class RandomTokenDetectionPipeline(Pipeline):
    r""" A model that use RTD loss where the probability of swapping each token is random. """

    POST_FORWARD_ADAPTER = token_detection_adaptation

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        metrics_kwargs = dict(average=None, ignore_index=IGNORE_IDX, num_classes=2)
        ppl_kwargs = dict(ignore_index=IGNORE_IDX)

        # train metrics
        self.train_acc = MulticlassAccuracy(**metrics_kwargs)
        self.train_f1 = MulticlassF1Score(**metrics_kwargs)
        self.train_ppl = Perplexity(**ppl_kwargs)

        # validation metrics
        self.valid_acc = MulticlassAccuracy(**metrics_kwargs)
        self.valid_f1 = MulticlassF1Score(**metrics_kwargs)
        self.valid_ppl = Perplexity(**ppl_kwargs)

        # test metrics
        self.test_acc = MulticlassAccuracy(**metrics_kwargs)
        self.test_f1 = MulticlassF1Score(**metrics_kwargs)
        self.test_ppl = Perplexity(**ppl_kwargs)

    def step(self, batch: Dict) -> TokenDetectionStepOutput:
        r""" Forward step is shared between all train/val/test steps. """
        results: TokenDetectionOutput = self.forward(**batch)

        # some models like ELECTRA outputs a single value per prediction
        if results.token_detection_logits.dim() == 2:
            results.token_detection_logits = expand_logits(results.token_detection_logits)

        return TokenDetectionStepOutput(
            loss=results.token_detection_loss,
            token_detection_loss=results.token_detection_loss,
            token_detection_predictions=results.token_detection_logits.argmax(dim=-1),
            token_detection_logits=results.token_detection_logits,
            token_detection_labels=batch['token_detection_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        train_acc = self.train_acc(step_output.token_detection_predictions, step_output.token_detection_labels)
        train_f1 = self.train_f1(step_output.token_detection_predictions, step_output.token_detection_labels)
        train_ppl = self.train_ppl(step_output.token_detection_logits.float(), step_output.token_detection_labels)

        self.log(LOSS, step_output.loss)
        self.log(TOKEN_DETECTION_LOSS, step_output.token_detection_loss)
        self.log(TOKEN_DETECTION_ACCURACY, train_acc)
        self.log(TOKEN_DETECTION_F1, train_f1)
        self.log(TOKEN_DETECTION_PERPLEXITY, train_ppl)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output = self.step(batch)

        valid_acc = self.valid_acc(step_output.token_detection_predictions, step_output.token_detection_labels)
        valid_f1 = self.valid_f1(step_output.token_detection_predictions, step_output.token_detection_labels)
        valid_ppl = self.valid_ppl(step_output.token_detection_logits.float(), step_output.token_detection_labels)

        self.log(LOSS, step_output.loss)
        self.log(TOKEN_DETECTION_LOSS, step_output.token_detection_loss)
        self.log(TOKEN_DETECTION_ACCURACY, valid_acc)
        self.log(TOKEN_DETECTION_F1, valid_f1)
        self.log(TOKEN_DETECTION_PERPLEXITY, valid_ppl)

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output = self.step(batch)
       
        test_acc = self.test_acc(step_output.token_detection_predictions, step_output.token_detection_labels)
        test_f1 = self.test_f1(step_output.token_detection_predictions, step_output.token_detection_labels)
        test_ppl = self.test_ppl(step_output.token_detection_logits.float(), step_output.token_detection_labels)

        self.log(LOSS, step_output.loss)
        self.log(TOKEN_DETECTION_LOSS, step_output.token_detection_loss)
        self.log(TOKEN_DETECTION_ACCURACY, test_acc)
        self.log(TOKEN_DETECTION_F1, test_f1)
        self.log(TOKEN_DETECTION_PERPLEXITY, test_ppl)

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add denoising objective. """
        return random_token_detection_processor(
            sample=sample,
            input_columns=self.hyperparameters.input_columns,
            probability=self.hyperparameters.probability,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            whole_word_detection=self.hyperparameters.whole_word_detection,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--input_columns', type=str, nargs='+', required=True)
        add_random_token_detection_arguments(parser)
