from typing import Any, Dict, List, Tuple, Union

import torch
from datasets import Dataset
from torchmetrics.classification.accuracy import MulticlassAccuracy
from transformers.configuration_utils import PretrainedConfig

from transformers_framework.architectures.modeling_outputs import QuestionAnsweringOutput
from transformers_framework.interfaces.adaptation import question_answering_adaptation
from transformers_framework.interfaces.logging import (
    LOSS,
    QUESTION_ANSWERING_END_ACCURACY,
    QUESTION_ANSWERING_EXACT_MATCH,
    QUESTION_ANSWERING_F1,
    QUESTION_ANSWERING_LOSS,
    QUESTION_ANSWERING_START_ACCURACY,
)
from transformers_framework.interfaces.step import QuestionAnsweringStepOutput
from transformers_framework.pipelines.pipeline import Pipeline
from transformers_framework.processing.postprocessors import question_answering_processor
from transformers_framework.processing.preprocessors import prepare_dataset_question_answering
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_question_answering_arguments
from transformers_framework.utilities.distributed import sync_data_distributed
from transformers_framework.utilities.evaluation import compute_question_answering_metrics


class QuestionAnsweringPipeline(Pipeline):

    POST_FORWARD_ADAPTER = question_answering_adaptation
    MODEL_INPUT_NAMES_TO_REDUCE = None

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        metrics_args = ()
        metrics_hparams = dict(num_classes=self.hyperparameters.max_sequence_length, average='micro')

        # train metrics
        self.train_start_acc = MulticlassAccuracy(*metrics_args, **metrics_hparams)
        self.train_end_acc = MulticlassAccuracy(*metrics_args, **metrics_hparams)

        # validation metrics
        self.validation_step_outputs: List[QuestionAnsweringStepOutput] = []
        self.valid_start_acc = MulticlassAccuracy(*metrics_args, **metrics_hparams)
        self.valid_end_acc = MulticlassAccuracy(*metrics_args, **metrics_hparams)

        # test metrics
        self.test_step_outputs: List[QuestionAnsweringStepOutput] = []
        self.test_start_acc = MulticlassAccuracy(*metrics_args, **metrics_hparams)
        self.test_end_acc = MulticlassAccuracy(*metrics_args, **metrics_hparams)

    def setup_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        kwargs['num_labels'] = 2  # always 2 classes for extractive question answering
        return super().setup_config(**kwargs)

    def step(self, batch: Dict) -> QuestionAnsweringStepOutput:
        r""" Forward step is shared between all train/val/test steps. """

        index = batch.pop('index')
        tokens = batch.pop('tokens')
        covered_tokens = batch.pop('covered_tokens')
        token_is_max_context = batch.pop('token_is_max_context')
        offset_mapping = batch.pop('offset_mapping')
        context = batch.pop('context')
        gold_answers = batch.pop('gold_answers')

        results: QuestionAnsweringOutput = self.forward(**batch)

        # start_logits and end_logits shape: (batch_size, seq_len)
        preds_start = results.start_position_logits.argmax(dim=-1)
        preds_end = results.end_position_logits.argmax(dim=-1)

        return QuestionAnsweringStepOutput(
            loss=results.question_answering_loss,
            question_answering_index=index,
            question_answering_loss=results.question_answering_loss,
            question_answering_start_logits=results.start_position_logits,
            question_answering_end_logits=results.end_position_logits,
            question_answering_start_predictions=preds_start,
            question_answering_end_predictions=preds_end,
            question_answering_start_labels=batch['start_position_labels'],
            question_answering_end_labels=batch['end_position_labels'],
            question_answering_tokens=tokens,
            question_answering_covered_tokens=covered_tokens,
            question_answering_token_is_max_context=token_is_max_context,
            question_answering_offset_mapping=offset_mapping,
            question_answering_context=context,
            question_answering_gold_answers=gold_answers,
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        train_start_acc = self.train_start_acc(
            step_output.question_answering_start_predictions, step_output.question_answering_start_labels
        )
        train_end_acc = self.train_end_acc(
            step_output.question_answering_end_predictions, step_output.question_answering_end_labels
        )

        self.log(LOSS, step_output.loss)
        self.log(QUESTION_ANSWERING_LOSS, step_output.question_answering_loss)
        self.log(QUESTION_ANSWERING_START_ACCURACY, train_start_acc)
        self.log(QUESTION_ANSWERING_END_ACCURACY, train_end_acc)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output = self.step(batch)

        valid_start_acc = self.valid_start_acc(
            step_output.question_answering_start_predictions, step_output.question_answering_start_labels
        )
        valid_end_acc = self.valid_end_acc(
            step_output.question_answering_end_predictions, step_output.question_answering_end_labels
        )

        self.log(LOSS, step_output.loss)
        self.log(QUESTION_ANSWERING_LOSS, step_output.question_answering_loss)
        self.log(QUESTION_ANSWERING_START_ACCURACY, valid_start_acc)
        self.log(QUESTION_ANSWERING_END_ACCURACY, valid_end_acc)

        self.validation_step_outputs.append(step_output)

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output = self.step(batch)

        test_start_acc = self.test_start_acc(
            step_output.question_answering_start_predictions, step_output.question_answering_start_labels
        )
        test_end_acc = self.test_end_acc(
            step_output.question_answering_end_predictions, step_output.question_answering_end_labels
        )

        self.log(LOSS, step_output.loss)
        self.log(QUESTION_ANSWERING_LOSS, step_output.question_answering_loss)
        self.log(QUESTION_ANSWERING_START_ACCURACY, test_start_acc)
        self.log(QUESTION_ANSWERING_END_ACCURACY, test_end_acc)

        self.test_step_outputs.append(step_output)

    def eval_epoch_end(self, outputs: List[QuestionAnsweringStepOutput]) -> Tuple:
        r""" Group alla predictions together and compute epoch level exact match and accuracy. """

        index = torch.cat([o.question_answering_index for o in outputs], dim=0)
        start_logits = torch.cat([o.question_answering_start_logits for o in outputs], dim=0)
        end_logits = torch.cat([o.question_answering_end_logits for o in outputs], dim=0)

        tokens = [x for o in outputs for x in o.question_answering_tokens]
        covered_tokens = [x for o in outputs for x in o.question_answering_covered_tokens]
        token_is_max_context = [x for o in outputs for x in o.question_answering_token_is_max_context]
        offset_mapping = [x for o in outputs for x in o.question_answering_offset_mapping]
        context = [x for o in outputs for x in o.question_answering_context]
        gold_answers = [x for o in outputs for x in o.question_answering_gold_answers]

        all_index = self.all_gather(index).view(-1).detach().cpu().tolist()
        all_start_logits = self.all_gather(start_logits).view(-1, start_logits.shape[-1]).detach().cpu().tolist()
        all_end_logits = self.all_gather(end_logits).view(-1, end_logits.shape[-1]).detach().cpu().tolist()

        all_tokens = sync_data_distributed(tokens, concat=True)
        all_covered_tokens = sync_data_distributed(covered_tokens, concat=True)
        all_token_is_max_context = sync_data_distributed(token_is_max_context, concat=True)
        all_offset_mapping = sync_data_distributed(offset_mapping, concat=True)
        all_context = sync_data_distributed(context, concat=True)
        all_gold_answers = sync_data_distributed(gold_answers, concat=True)

        exact_match, f1 = compute_question_answering_metrics(
            all_index=all_index,
            all_start_logits=all_start_logits,
            all_end_logits=all_end_logits,
            all_tokens=all_tokens,
            all_covered_tokens=all_covered_tokens,
            all_token_is_max_context=all_token_is_max_context,
            all_offset_mapping=all_offset_mapping,
            all_context=all_context,
            all_gold_answers=all_gold_answers,
            n_best_size=self.hyperparameters.n_best_size,
            max_answer_length=self.hyperparameters.max_answer_length,
        )

        return exact_match, f1

    def on_validation_epoch_end(self):
        r""" Compute and log global EM and F1 for machine reading. """
        super().on_validation_epoch_end()
        exact_match, f1 = self.eval_epoch_end(outputs=self.validation_step_outputs)

        self.log(QUESTION_ANSWERING_EXACT_MATCH, exact_match)
        self.log(QUESTION_ANSWERING_F1, f1)

    def on_test_epoch_end(self):
        r""" Compute and log global EM and F1 for machine reading. """
        super().on_test_epoch_end()
        exact_match, f1 = self.eval_epoch_end(outputs=self.test_step_outputs)

        self.log(QUESTION_ANSWERING_EXACT_MATCH, exact_match)
        self.log(QUESTION_ANSWERING_F1, f1)

    def preprocess(
        self, dataset: Dataset, num_workers: int = None, batch_size: int = None, load_from_cache_file: bool = True
    ) -> Dataset:
        r""" Preprocess dataset for machine reading """
        return prepare_dataset_question_answering(
            dataset,
            query_column=self.hyperparameters.query_column,
            context_column=self.hyperparameters.context_column,
            answers_column=self.hyperparameters.answers_column,
            label_column=self.hyperparameters.label_column,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            doc_stride=self.hyperparameters.doc_stride,
            max_query_length=self.hyperparameters.max_query_length,
            tokenizer=self.tokenizer,
            load_from_cache_file=load_from_cache_file,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Convert model inputs to numpy. """
        return question_answering_processor(sample=sample)

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument(
            '--query_column', required=False, type=str, default='question', help="Column name of questions"
        )
        parser.add_argument(
            '--context_column', required=False, type=str, default='context', help="Column name of context"
        )
        parser.add_argument(
            '--answers_column', required=False, type=str, default='answers', help="Column name of answers"
        )
        parser.add_argument(
            '--label_column', required=False, type=str, default='labels', help="Column name of labels"
        )
        add_question_answering_arguments(parser)
