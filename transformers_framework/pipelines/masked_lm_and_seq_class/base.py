from typing import Any, Dict, Union

from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score
from transformers.configuration_utils import PretrainedConfig

from transformers_framework.architectures.modeling_outputs import MaskedLMAndSeqClassOutput
from transformers_framework.interfaces.logging import (
    LOSS,
    MASKED_LM_ACCURACY,
    MASKED_LM_LOSS,
    MASKED_LM_PERPLEXITY,
    SEQ_CLASS_ACCURACY,
    SEQ_CLASS_F1,
    SEQ_CLASS_LOSS,
)
from transformers_framework.interfaces.step import MaskedLMAndSeqClassStepOutput
from transformers_framework.metrics.perplexity import Perplexity
from transformers_framework.pipelines.pipeline.pipeline import ExtendedPipeline
from transformers_framework.processing.postprocessors import masked_lm_and_seq_class_processor
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import (
    FlexibleArgumentParser,
    add_masked_lm_arguments,
    add_seq_class_arguments,
)
from transformers_framework.utilities.torch import combine_losses


class MaskedLMAndSeqClassPipeline(ExtendedPipeline):

    POST_FORWARD_ADAPTER = None  # models designed from MLM + Classification return already a compatible object
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels')
    ]

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        # needed till full support for multiple classification is implemented
        assert self.hyperparameters.k is None

        metrics_args = (self.hyperparameters.num_labels, )
        metrics_kwargs = dict(average=None, ignore_index=IGNORE_IDX)

        # train metrics
        self.train_mlm_acc = MulticlassAccuracy(self.tokenizer.vocab_size, average='micro', ignore_index=IGNORE_IDX)
        self.train_mlm_ppl = Perplexity(ignore_index=IGNORE_IDX)
        self.train_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.train_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

        # validation metrics
        self.valid_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.valid_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

        # test metrics
        self.test_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.test_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

    def requires_extended_tokenizer(self):
        return len(self.hyperparameters.input_columns) > 2

    def requires_extended_model(self):
        return self.hyperparameters.k is not None

    def configure_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        kwargs['num_labels'] = self.hyperparameters.num_labels
        return super().configure_config(**kwargs)

    def step(self, batch: Dict) -> MaskedLMAndSeqClassStepOutput:
        r""" Forward step is shared between all train/val/test steps. """

        results: MaskedLMAndSeqClassOutput = self.forward(**batch)

        loss = combine_losses(
            losses=[results.masked_lm_loss, results.seq_class_loss],
            weigths=[self.hyperparameters.masked_lm_weight, self.hyperparameters.seq_class_weight],
        )

        return MaskedLMAndSeqClassStepOutput(
            loss=loss,
            masked_lm_loss=results.masked_lm_loss,
            masked_lm_predictions=results.masked_lm_logits.argmax(dim=-1),
            masked_lm_logits=results.masked_lm_logits,
            masked_lm_labels=batch['masked_lm_labels'],
            seq_class_loss=results.seq_class_loss,
            seq_class_predictions=results.seq_class_logits.argmax(dim=-1),
            seq_class_labels=batch['seq_class_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        train_mlm_acc = self.train_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        train_mlm_ppl = self.train_mlm_ppl(step_output.masked_lm_logits.float(), step_output.masked_lm_labels)
        train_acc = self.train_acc(step_output.seq_class_predictions, step_output.seq_class_labels)
        train_f1 = self.train_f1(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(MASKED_LM_LOSS, step_output.masked_lm_loss)
        self.log(MASKED_LM_ACCURACY, train_mlm_acc)
        self.log(MASKED_LM_PERPLEXITY, train_mlm_ppl)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, train_acc)
        self.log(SEQ_CLASS_F1, train_f1)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output = self.step(batch)

        valid_acc = self.valid_acc(step_output.seq_class_predictions, step_output.seq_class_labels)
        valid_f1 = self.valid_f1(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, valid_acc)
        self.log(SEQ_CLASS_F1, valid_f1)

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output = self.step(batch)

        test_acc = self.test_acc(step_output.seq_class_predictions, step_output.seq_class_labels)
        test_f1 = self.test_f1(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, test_acc)
        self.log(SEQ_CLASS_F1, test_f1)

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add denoising objective. """
        return masked_lm_and_seq_class_processor(
            sample=sample,
            input_columns=self.hyperparameters.input_columns,
            label_column=self.hyperparameters.label_column,
            probability=self.hyperparameters.probability,
            probability_masked=self.hyperparameters.probability_masked,
            probability_replaced=self.hyperparameters.probability_replaced,
            probability_unchanged=self.hyperparameters.probability_unchanged,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            whole_word_masking=self.hyperparameters.whole_word_masking,
            training=self.training,
            extended_token_type_ids=self.hyperparameters.extended_token_type_ids,
            k=self.hyperparameters.k,
            return_original_input_ids=False,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        add_seq_class_arguments(parser)
        add_masked_lm_arguments(parser)
        parser.add_argument('--input_columns', type=str, nargs='+', required=True)
        parser.add_argument('--label_column', type=str, required=True)
        parser.add_argument('--seq_class_weight', type=float, default=1.0)
        parser.add_argument('--masked_lm_weight', type=float, default=1.0)
