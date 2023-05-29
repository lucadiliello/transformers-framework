from typing import Any, Dict, Union

from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers_framework.architectures.modeling_outputs import TokenClassOutput
from transformers_framework.interfaces.adaptation import token_classification_adaptation
from transformers_framework.interfaces.logging import (
    LOSS,
    TOKEN_CLASS_ACCURACY,
    TOKEN_CLASS_F1,
    TOKEN_CLASS_LOSS,
    TOKEN_CLASS_PERPLEXITY,
)
from transformers_framework.interfaces.step import TokenClassStepOutput
from transformers_framework.metrics.perplexity import Perplexity
from transformers_framework.pipelines.pipeline.pipeline import Pipeline
from transformers_framework.processing.postprocessors import token_class_processor
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_token_class_arguments


class TokenClassPipeline(Pipeline):
    r""" A model that does token classification for NER or POS. """

    POST_FORWARD_ADAPTER = token_classification_adaptation
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'attention_mask', 'token_type_ids', 'token_class_labels')
    ]

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        metrics_kwargs = dict(average=None, ignore_index=IGNORE_IDX, num_classes=self.config.num_labels)
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

    def configure_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        kwargs['num_labels'] = self.hyperparameters.num_labels
        return super().configure_config(**kwargs)

    def configure_tokenizer(self) -> PreTrainedTokenizerBase:
        return super().configure_tokenizer(add_prefix_space=True)

    def step(self, batch: Dict) -> TokenClassStepOutput:
        r""" Forward step is shared between all train/val/test steps. """
        results: TokenClassOutput = self.forward(**batch)

        return TokenClassStepOutput(
            loss=results.token_class_loss,
            token_class_loss=results.token_class_loss,
            token_class_predictions=results.token_class_logits.argmax(dim=-1),
            token_class_logits=results.token_class_logits,
            token_class_labels=batch['token_class_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        acc = self.train_acc(step_output.token_class_predictions, step_output.token_class_labels)
        f1 = self.train_f1(step_output.token_class_predictions, step_output.token_class_labels)
        ppl = self.train_ppl(step_output.token_class_logits.float(), step_output.token_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(TOKEN_CLASS_LOSS, step_output.token_class_loss)
        self.log(TOKEN_CLASS_ACCURACY, acc)
        self.log(TOKEN_CLASS_PERPLEXITY, ppl)
        self.log(TOKEN_CLASS_F1, f1)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output = self.step(batch)

        valid_acc = self.valid_acc(
            step_output.token_class_predictions, step_output.token_class_labels
        )
        valid_f1 = self.valid_f1(step_output.token_class_predictions, step_output.token_class_labels)
        valid_ppl = self.valid_ppl(step_output.token_class_logits.float(), step_output.token_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(TOKEN_CLASS_LOSS, step_output.token_class_loss)
        self.log(TOKEN_CLASS_ACCURACY, valid_acc)
        self.log(TOKEN_CLASS_F1, valid_f1)
        self.log(TOKEN_CLASS_PERPLEXITY, valid_ppl)

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output = self.step(batch)
       
        test_acc = self.test_acc(step_output.token_class_predictions, step_output.token_class_labels)
        test_f1 = self.test_f1(step_output.token_class_predictions, step_output.token_class_labels)
        test_ppl = self.test_ppl(step_output.token_class_logits.float(), step_output.token_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(TOKEN_CLASS_LOSS, step_output.token_class_loss)
        self.log(TOKEN_CLASS_ACCURACY, test_acc)
        self.log(TOKEN_CLASS_F1, test_f1)
        self.log(TOKEN_CLASS_PERPLEXITY, test_ppl)

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add denoising objective. """
        return token_class_processor(
            sample=sample,
            input_column=self.hyperparameters.input_column,
            label_column=self.hyperparameters.label_column,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--input_column', type=str, required=True)
        parser.add_argument('--label_column', type=str, required=True)
        add_token_class_arguments(parser)
