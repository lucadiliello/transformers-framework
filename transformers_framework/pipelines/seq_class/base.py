from typing import Any, Dict, Union

from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score
from transformers.configuration_utils import PretrainedConfig

from transformers_framework.interfaces.adaptation import sequence_classification_adaptation
from transformers_framework.interfaces.logging import LOSS, SEQ_CLASS_ACCURACY, SEQ_CLASS_F1
from transformers_framework.interfaces.step import SeqClassStepOutput
from transformers_framework.pipelines.pipeline import ExtendedPipeline
from transformers_framework.processing.postprocessors import seq_class_processor
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_seq_class_arguments


class SeqClassPipeline(ExtendedPipeline):

    POST_FORWARD_ADAPTER = sequence_classification_adaptation
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'attention_mask', 'token_type_ids')
    ]

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        # needed till full support for multiple classification is implemented
        assert self.hyperparameters.k is None

        metrics_args = (self.hyperparameters.num_labels, )
        metrics_kwargs = dict(average=None, ignore_index=IGNORE_IDX)

        # train metrics
        self.train_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.train_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

        # validation metrics
        self.valid_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.valid_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

        # test metrics
        self.test_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.test_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

    def setup_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        kwargs['num_labels'] = self.hyperparameters.num_labels
        return super().setup_config(**kwargs)

    def step(self, batch: Dict) -> SeqClassStepOutput:
        r""" Forward step is shared between all train/val/test steps. """
        results = self.forward(**batch)
        predictions = results.seq_class_logits.argmax(dim=-1)

        return SeqClassStepOutput(
            loss=results.seq_class_loss,
            seq_class_loss=results.seq_class_loss,
            seq_class_predictions=predictions,
            seq_class_labels=batch['seq_class_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        train_acc = self.train_acc(step_output.seq_class_predictions, step_output.seq_class_labels)
        train_f1 = self.train_f1(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
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
        self.log(SEQ_CLASS_ACCURACY, test_acc)
        self.log(SEQ_CLASS_F1, test_f1)

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Sample should contain question, answer and optionally other columns that will be encoded together. """
        return seq_class_processor(
            sample=sample,
            input_columns=self.hyperparameters.input_columns,
            label_column=self.hyperparameters.label_column,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            extended_token_type_ids=self.hyperparameters.extended_token_type_ids,
            k=self.hyperparameters.k,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--input_columns', type=str, nargs='+', required=True)
        parser.add_argument('--label_column', type=str, required=True)
        add_seq_class_arguments(parser)
