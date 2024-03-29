from types import MethodType
from typing import Any, Dict, Union

import torch
from datasets import Dataset
from torchmetrics.classification.accuracy import BinaryAccuracy
from torchmetrics.classification.f_beta import BinaryF1Score
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
)
from transformers.configuration_utils import PretrainedConfig

from transformers_framework.architectures.modeling_outputs import SeqClassOutput
from transformers_framework.interfaces.adaptation import sequence_classification_adaptation
from transformers_framework.interfaces.logging import (
    ANSWER_SELECTION_HR,
    ANSWER_SELECTION_MAP,
    ANSWER_SELECTION_MRR,
    ANSWER_SELECTION_NDCG,
    ANSWER_SELECTION_PRECISION,
    LOSS,
    SEQ_CLASS_ACCURACY,
    SEQ_CLASS_F1,
    SEQ_CLASS_LOSS,
)
from transformers_framework.interfaces.step import AnswerSelectionStepOutput
from transformers_framework.pipelines.pipeline import ExtendedPipeline
from transformers_framework.processing.postprocessors import answer_selection_processor
from transformers_framework.processing.preprocessors import answer_selection_grouping
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_answer_selection_arguments


class AnswerSelectionPipeline(ExtendedPipeline):

    POST_FORWARD_ADAPTER = sequence_classification_adaptation
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'attention_mask', 'token_type_ids')
    ]

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        # check that when grouping is random, reload_dataloaders_every_n_epochs is 1
        if self.requires_extended_model:
            if self.hyperparameters.grouping == 'random' and not self.hyperparameters.reload_train_dataset_every_epoch:
                raise ValueError("You must specify `reload_dataloaders_every_n_epochs=1` when using `grouping=random`")

        self.label_weights = None
        if self.hyperparameters.label_weights is not None:
            assert len(self.hyperparameters.label_weights) == 2
            self.label_weights = torch.tensor(self.hyperparameters.label_weights)

        # train metrics
        self.train_acc = BinaryAccuracy(ignore_index=IGNORE_IDX)
        self.train_f1 = BinaryF1Score(ignore_index=IGNORE_IDX)

        # validation metrics
        metrics_kwargs = dict(
            empty_target_action=hyperparameters.metrics_empty_target_action,
            ignore_index=IGNORE_IDX,
        )
        self.valid_acc = BinaryAccuracy(ignore_index=IGNORE_IDX)
        self.valid_f1 = BinaryF1Score(ignore_index=IGNORE_IDX)
        self.valid_map = RetrievalMAP(**metrics_kwargs)
        self.valid_mrr = RetrievalMRR(**metrics_kwargs)
        self.valid_p1 = RetrievalPrecision(top_k=1, **metrics_kwargs)
        self.valid_hr5 = RetrievalHitRate(top_k=5, **metrics_kwargs)
        self.valid_ndgc = RetrievalNormalizedDCG(**metrics_kwargs)

        # test metrics
        self.test_acc = BinaryAccuracy(ignore_index=IGNORE_IDX)
        self.test_f1 = BinaryF1Score(ignore_index=IGNORE_IDX)
        self.test_map = RetrievalMAP(**metrics_kwargs)
        self.test_mrr = RetrievalMRR(**metrics_kwargs)
        self.test_p1 = RetrievalPrecision(top_k=1, **metrics_kwargs)
        self.test_hr5 = RetrievalHitRate(top_k=5, **metrics_kwargs)
        self.test_ndgc = RetrievalNormalizedDCG(**metrics_kwargs)

        if self.requires_extended_model:
            self.preprocess = MethodType(preprocess, self)

        if self.hyperparameters.separated:
            self.MODEL_INPUT_NAMES_TO_REDUCE = None  # do not shrink batches when using separated mode (joint models)

    def setup_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        kwargs['num_labels'] = 2  # always 2 classes for answer selection
        return super().setup_config(**kwargs)

    def step(self, batch: Dict) -> AnswerSelectionStepOutput:
        r""" Forward step is shared between all train/val/test steps. """

        if self.label_weights is not None:
            batch['weights'] = self.label_weights.to(device=self.device)

        index = batch.pop('index')
        results: SeqClassOutput = self.forward(**batch)

        # reranking
        if self.hyperparameters.k is not None:
            assert results.seq_class_logits.dim() == 3 and results.seq_class_logits.shape[1] == self.hyperparameters.k
            assert batch['seq_class_labels'].dim() == 2 and batch['seq_class_labels'].shape[1] == self.hyperparameters.k
            # [batch size, k, num_labels], num_labels = 2
            results.seq_class_logits = results.seq_class_logits[batch['seq_class_labels'] != -1]

        assert results.seq_class_logits.dim() == 2  # [batch size, num_labels], num_labels = 2
        seq_class_scores = results.seq_class_logits.softmax(dim=-1)[:, -1]
        seq_class_predictions = results.seq_class_logits.argmax(dim=-1)

        index = index.flatten()
        batch['seq_class_labels'] = batch['seq_class_labels'].flatten()

        return AnswerSelectionStepOutput(
            loss=results.seq_class_loss,
            seq_class_index=index,
            seq_class_loss=results.seq_class_loss,
            seq_class_predictions=seq_class_predictions,
            seq_class_scores=seq_class_scores,
            seq_class_labels=batch['seq_class_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        train_acc = self.train_acc(step_output.seq_class_predictions, step_output.seq_class_labels)
        train_f1 = self.train_f1(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, train_acc)
        self.log(SEQ_CLASS_F1, train_f1)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output = self.step(batch)

        # AS2 metrics should only be computed globally
        self.validation_step_update_metrics(step_output)

        # logging
        val_acc = self.valid_acc(step_output.seq_class_predictions, step_output.seq_class_labels)
        val_f1 = self.valid_f1(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, val_acc)
        self.log(SEQ_CLASS_F1, val_f1)

    def validation_step_update_metrics(self, step_output: AnswerSelectionStepOutput):
        r""" Update metrics for answer selection. """
        kwargs = dict(
            preds=step_output.seq_class_scores,
            target=step_output.seq_class_labels,
            indexes=step_output.seq_class_index,
        )
        self.valid_map.update(**kwargs)
        self.valid_mrr.update(**kwargs)
        self.valid_p1.update(**kwargs)
        self.valid_hr5.update(**kwargs)
        self.valid_ndgc.update(**kwargs)

    def test_step(self, batch, *args):
        r""" Compute predictions and log retrieval results. """
        step_output = self.step(batch)

        # AS2 metrics should only be computed globally
        self.test_step_update_metrics(step_output)

        # logging
        test_acc = self.test_acc(step_output.seq_class_predictions, step_output.seq_class_labels)
        test_f1 = self.test_f1(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, test_acc)
        self.log(SEQ_CLASS_F1, test_f1)

    def test_step_update_metrics(self, step_output: AnswerSelectionStepOutput):
        r""" Update metrics for answer selection. """
        kwargs = dict(
            preds=step_output.seq_class_scores,
            target=step_output.seq_class_labels,
            indexes=step_output.seq_class_index,
        )
        self.test_map.update(**kwargs)
        self.test_mrr.update(**kwargs)
        self.test_p1.update(**kwargs)
        self.test_hr5.update(**kwargs)
        self.test_ndgc.update(**kwargs)

    def on_validation_epoch_end(self):
        r""" Just log metrics. """
        super().on_validation_epoch_end()
        self.log(ANSWER_SELECTION_MAP, self.valid_map.compute())
        self.log(ANSWER_SELECTION_MRR, self.valid_mrr.compute())
        self.log(ANSWER_SELECTION_PRECISION(1), self.valid_p1.compute())
        self.log(ANSWER_SELECTION_HR(5), self.valid_hr5.compute())
        self.log(ANSWER_SELECTION_NDCG, self.valid_ndgc.compute())

    def on_test_epoch_end(self):
        r""" Just log metrics. """
        super().on_test_epoch_end()
        self.log(ANSWER_SELECTION_MAP, self.test_map.compute())
        self.log(ANSWER_SELECTION_MRR, self.test_mrr.compute())
        self.log(ANSWER_SELECTION_PRECISION(1), self.test_p1.compute())
        self.log(ANSWER_SELECTION_HR(5), self.test_hr5.compute())
        self.log(ANSWER_SELECTION_NDCG, self.test_ndgc.compute())

    def predict_step(self, batch, *args):
        r""" Predict and return scores. """
        return self.step(batch).seq_class_scores

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Sample should contain question, answer and optionally other columns that will be encoded together. """
        return answer_selection_processor(
            sample=sample,
            input_columns=self.hyperparameters.input_columns,
            index_column=self.hyperparameters.index_column,
            label_column=self.hyperparameters.label_column,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            extended_token_type_ids=self.hyperparameters.extended_token_type_ids,
            k=self.hyperparameters.k,
            separated=self.hyperparameters.separated,
            pad_to_k=self.hyperparameters.pad_to_k,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--input_columns', type=str, nargs='+', required=True)
        parser.add_argument('--label_column', type=str, required=True)
        parser.add_argument('--index_column', type=str, required=False)
        parser.add_argument('--label_weights', type=float, nargs=2, required=False, default=None)
        add_answer_selection_arguments(parser)
        parser.add_argument('--separated', action="store_true")
        parser.add_argument_if_not('k', None, '--grouping', type=str, required=True, choices=('fixed', 'random'))
        parser.add_argument_if_not(
            'k', None, '--selection', type=str, required=False, choices=('best', 'worst'), default=None
        )
        parser.add_argument_if_not('k', None, '--scores_column', type=str, required=False, default=None)


def preprocess(
    self: AnswerSelectionPipeline,
    dataset: Dataset,
    num_workers: int = None,
    batch_size: int = None,
    load_from_cache_file: bool = True,
) -> Dataset:
    r""" Preprocess a dataset. Useful for tasks like Extended Classification. """
    return answer_selection_grouping(
        dataset=dataset,
        input_columns=self.hyperparameters.input_columns,
        index_column=self.hyperparameters.index_column,
        label_column=self.hyperparameters.label_column,
        k=self.hyperparameters.k,
        num_workers=num_workers,
        batch_size=batch_size,
        grouping=self.hyperparameters.grouping,
        selection=self.hyperparameters.selection,
        scores_column=self.hyperparameters.scores_column,
        load_from_cache_file=load_from_cache_file,
    )
