from typing import Any, Dict

from torchmetrics.classification.accuracy import MulticlassAccuracy

from transformers_framework.architectures.modeling_outputs import MaskedLMOutput
from transformers_framework.interfaces.adaptation import masked_lm_adaptation
from transformers_framework.interfaces.logging import MASKED_LM_ACCURACY, MASKED_LM_LOSS, MASKED_LM_PERPLEXITY
from transformers_framework.interfaces.step import MaskedLMStepOutput
from transformers_framework.metrics.perplexity import Perplexity
from transformers_framework.pipelines.pipeline import Pipeline
from transformers_framework.processing.postprocessors import masked_lm_processor
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_masked_lm_arguments


class MaskedLMPipeline(Pipeline):

    POST_FORWARD_ADAPTER = masked_lm_adaptation
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels')
    ]

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        metrics_args = (self.tokenizer.vocab_size, )
        metrics_kwargs = dict(average='micro', ignore_index=IGNORE_IDX)

        # accuracies
        self.train_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.valid_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.test_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)

        # perplexities
        metrics_kwargs = dict(ignore_index=IGNORE_IDX)
        self.train_mlm_ppl = Perplexity(**metrics_kwargs)
        self.valid_mlm_ppl = Perplexity(**metrics_kwargs)
        self.test_mlm_ppl = Perplexity(**metrics_kwargs)

    def step(self, batch: Dict) -> MaskedLMStepOutput:
        r""" Forward step is shared between all train/val/test steps. """
        results: MaskedLMOutput = self.forward(**batch)

        return MaskedLMStepOutput(
            loss=results.masked_lm_loss,
            masked_lm_loss=results.masked_lm_loss,
            masked_lm_predictions=results.masked_lm_logits.argmax(dim=-1),
            masked_lm_logits=results.masked_lm_logits,
            masked_lm_labels=batch['masked_lm_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        train_mlm_acc = self.train_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        train_mlm_ppl = self.train_mlm_ppl(step_output.masked_lm_logits.float(), step_output.masked_lm_labels)

        self.log(MASKED_LM_LOSS, step_output.loss)
        self.log(MASKED_LM_ACCURACY, train_mlm_acc)
        self.log(MASKED_LM_PERPLEXITY, train_mlm_ppl)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output = self.step(batch)

        valid_mlm_acc = self.valid_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        valid_mlm_ppl = self.valid_mlm_ppl(step_output.masked_lm_logits.float(), step_output.masked_lm_labels)

        self.log(MASKED_LM_LOSS, step_output.loss)
        self.log(MASKED_LM_ACCURACY, valid_mlm_acc)
        self.log(MASKED_LM_PERPLEXITY, valid_mlm_ppl)

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output = self.step(batch)

        test_mlm_acc = self.test_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        test_mlm_ppl = self.test_mlm_ppl(step_output.masked_lm_logits.float(), step_output.masked_lm_labels)

        self.log(MASKED_LM_LOSS, step_output.loss)
        self.log(MASKED_LM_ACCURACY, test_mlm_acc)
        self.log(MASKED_LM_PERPLEXITY, test_mlm_ppl)

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add MLM objective. """
        return masked_lm_processor(
            sample=sample,
            input_columns=self.hyperparameters.input_columns,
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
        parser.add_argument(
            '--input_columns', type=str, nargs='+', required=True, help="Input column for training text"
        )
        add_masked_lm_arguments(parser)
