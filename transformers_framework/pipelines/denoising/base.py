from typing import Dict

from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score

from transformers_framework.architectures.modeling_outputs import SeqToSeqLMOutput
from transformers_framework.interfaces.adaptation import seq_to_seq_lm_adaptation
from transformers_framework.interfaces.logging import SEQ_TO_SEQ_LM_ACCURACY, SEQ_TO_SEQ_LM_F1, SEQ_TO_SEQ_LM_LOSS
from transformers_framework.interfaces.step import SeqToSeqMaskedLMStepOutput
from transformers_framework.pipelines.pipeline import Pipeline
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_denoising_arguments


class DenoisingPipeline(Pipeline):

    POST_FORWARD_ADAPTER = seq_to_seq_lm_adaptation
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'attention_mask', 'token_type_ids'),
        ('decoder_input_ids', 'decoder_attention_mask', 'decoder_token_type_ids', 'seq_to_seq_lm_labels'),
    ]

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        metrics_args = (self.tokenizer.vocab_size, )
        metrics_kwargs = dict(average='micro', ignore_index=self.tokenizer.pad_token_id)

        # train metrics
        self.train_denoising_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.train_denoising_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

        # validation metrics
        self.valid_denoising_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.valid_denoising_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

        # test metrics
        self.test_denoising_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.test_denoising_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

    def step(self, batch: Dict) -> SeqToSeqMaskedLMStepOutput:
        r""" Forward step is shared between all train/val/test steps. """
        results: SeqToSeqLMOutput = self.forward(**batch)

        return SeqToSeqMaskedLMStepOutput(
            loss=results.seq_to_seq_lm_loss,
            seq_to_seq_lm_loss=results.seq_to_seq_lm_loss,
            seq_to_seq_lm_predictions=results.seq_to_seq_lm_logits.argmax(dim=-1),
            seq_to_seq_lm_labels=batch['seq_to_seq_lm_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        train_denoising_acc = self.train_denoising_acc(
            step_output.seq_to_seq_lm_predictions, step_output.seq_to_seq_lm_labels
        )
        train_denoising_f1 = self.train_denoising_f1(
            step_output.seq_to_seq_lm_predictions, step_output.seq_to_seq_lm_labels
        )

        self.log(SEQ_TO_SEQ_LM_LOSS, step_output.loss)
        self.log(SEQ_TO_SEQ_LM_ACCURACY, train_denoising_acc)
        self.log(SEQ_TO_SEQ_LM_F1, train_denoising_f1)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output = self.step(batch)
        valid_denoising_acc = self.valid_denoising_acc(
            step_output.seq_to_seq_lm_predictions, step_output.seq_to_seq_lm_labels
        )
        valid_denoising_f1 = self.valid_denoising_f1(
            step_output.seq_to_seq_lm_predictions, step_output.seq_to_seq_lm_labels
        )

        self.log(SEQ_TO_SEQ_LM_LOSS, step_output.loss)
        self.log(SEQ_TO_SEQ_LM_ACCURACY, valid_denoising_acc)
        self.log(SEQ_TO_SEQ_LM_F1, valid_denoising_f1)

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output = self.step(batch)
        test_denoising_acc = self.test_denoising_acc(
            step_output.seq_to_seq_lm_predictions, step_output.seq_to_seq_lm_labels
        )
        test_denoising_f1 = self.test_denoising_f1(
            step_output.seq_to_seq_lm_predictions, step_output.seq_to_seq_lm_labels
        )

        self.log(SEQ_TO_SEQ_LM_LOSS, step_output.loss)
        self.log(SEQ_TO_SEQ_LM_ACCURACY, test_denoising_acc)
        self.log(SEQ_TO_SEQ_LM_F1, test_denoising_f1)

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        add_denoising_arguments(parser)
        parser.add_argument(
            '--input_column', type=str, required=True, help="Input column for training text"
        )
