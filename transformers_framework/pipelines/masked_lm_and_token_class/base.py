from typing import Any, Dict, Union

from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers_framework.architectures.modeling_outputs import MaskedLMAndTokenClassOutput
from transformers_framework.interfaces.adaptation import masked_lm_and_token_detection_adaptation
from transformers_framework.interfaces.logging import (
    LOSS,
    MASKED_LM_ACCURACY,
    MASKED_LM_LOSS,
    MASKED_LM_PERPLEXITY,
    TOKEN_CLASS_ACCURACY,
    TOKEN_CLASS_F1,
    TOKEN_CLASS_LOSS,
)
from transformers_framework.interfaces.step import MaskedLMAndTokenClassStepOutput
from transformers_framework.metrics.perplexity import Perplexity
from transformers_framework.pipelines.pipeline.pipeline import Pipeline
from transformers_framework.processing.postprocessors import masked_lm_and_token_class_processor
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import (
    FlexibleArgumentParser,
    add_masked_lm_arguments,
    add_token_class_arguments,
)
from transformers_framework.utilities.torch import combine_losses


class MaskedLMAndTokenClass(Pipeline):

    POST_FORWARD_ADAPTER = masked_lm_and_token_detection_adaptation
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels', 'token_class_labels')
    ]

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        # masked lm metrics
        metrics_args = (self.tokenizer.vocab_size, )
        metrics_kwargs = dict(average='micro', ignore_index=IGNORE_IDX)

        self.train_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.valid_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.test_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)

        metrics_kwargs = dict(ignore_index=IGNORE_IDX)
        self.train_mlm_ppl = Perplexity(**metrics_kwargs)
        self.valid_mlm_ppl = Perplexity(**metrics_kwargs)
        self.test_mlm_ppl = Perplexity(**metrics_kwargs)

        # classification metrics
        metrics_kwargs = dict(average=None, ignore_index=IGNORE_IDX, num_classes=self.config.num_labels)

        # train metrics
        self.train_acc = MulticlassAccuracy(**metrics_kwargs)
        self.train_f1 = MulticlassF1Score(**metrics_kwargs)

        # validation metrics
        self.valid_acc = MulticlassAccuracy(**metrics_kwargs)
        self.valid_f1 = MulticlassF1Score(**metrics_kwargs)

        # test metrics
        self.test_acc = MulticlassAccuracy(**metrics_kwargs)
        self.test_f1 = MulticlassF1Score(**metrics_kwargs)

    def configure_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        kwargs['num_labels'] = self.hyperparameters.num_labels
        return super().configure_config(**kwargs)

    def configure_tokenizer(self) -> PreTrainedTokenizerBase:
        return super().configure_tokenizer(add_prefix_space=True)

    def step(self, batch: Dict) -> MaskedLMAndTokenClassStepOutput:
        r""" Forward step is shared between all train/val/test steps. """

        results: MaskedLMAndTokenClassOutput = self.forward(**batch)

        # classification
        classification_predictions = results.token_class_logits.argmax(dim=-1)

        loss = combine_losses(
            losses=[results.masked_lm_loss, results.token_class_loss],
            weigths=[self.hyperparameters.masked_lm_weight, self.hyperparameters.token_class_weight],
        )

        return MaskedLMAndTokenClassStepOutput(
            loss=loss,
            masked_lm_loss=results.masked_lm_loss,
            masked_lm_predictions=results.masked_lm_logits.argmax(dim=-1),
            masked_lm_logits=results.masked_lm_logits,
            masked_lm_labels=batch['masked_lm_labels'],
            token_class_loss=results.token_class_loss,
            token_class_predictions=classification_predictions,
            token_class_logits=results.token_class_logits,
            token_class_labels=batch['token_class_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        train_mlm_acc = self.train_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        train_mlm_ppl = self.train_mlm_ppl(step_output.masked_lm_logits.float(), step_output.masked_lm_labels)

        train_acc = self.train_acc(
            step_output.token_class_predictions, step_output.token_class_labels
        )
        train_f1 = self.train_f1(step_output.token_class_predictions, step_output.token_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(MASKED_LM_LOSS, step_output.masked_lm_loss)
        self.log(MASKED_LM_ACCURACY, train_mlm_acc)
        self.log(MASKED_LM_PERPLEXITY, train_mlm_ppl)
        self.log(TOKEN_CLASS_LOSS, step_output.token_class_loss)
        self.log(TOKEN_CLASS_ACCURACY, train_acc)
        self.log(TOKEN_CLASS_F1, train_f1)

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

        self.log(LOSS, step_output.loss)
        self.log(TOKEN_CLASS_LOSS, step_output.token_class_loss)
        self.log(TOKEN_CLASS_ACCURACY, valid_acc)
        self.log(TOKEN_CLASS_F1, valid_f1)

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output = self.step(batch)

        test_acc = self.test_acc(
            step_output.token_class_predictions, step_output.token_class_labels
        )
        test_f1 = self.test_f1(step_output.token_class_predictions, step_output.token_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(TOKEN_CLASS_LOSS, step_output.token_class_loss)
        self.log(TOKEN_CLASS_ACCURACY, test_acc)
        self.log(TOKEN_CLASS_F1, test_f1)

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add denoising objective. """
        return masked_lm_and_token_class_processor(
            sample=sample,
            input_column=self.hyperparameters.input_column,
            label_column=self.hyperparameters.label_column,
            probability=self.hyperparameters.probability,
            probability_masked=self.hyperparameters.probability_masked,
            probability_replaced=self.hyperparameters.probability_replaced,
            probability_unchanged=self.hyperparameters.probability_unchanged,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            whole_word_masking=self.hyperparameters.whole_word_masking,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--input_column', type=str, required=True)
        parser.add_argument('--label_column', type=str, required=True)
        parser.add_argument('--token_class_weight', type=float, default=1.0)
        parser.add_argument('--masked_lm_weight', type=float, default=1.0)
        add_masked_lm_arguments(parser)
        add_token_class_arguments(parser)
