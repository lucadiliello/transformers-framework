from typing import Any, Dict, List

from torch import Tensor, cosine_similarity
from torchmetrics.classification.accuracy import BinaryAccuracy
from torchmetrics.classification.f_beta import BinaryF1Score
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP, RetrievalMRR, RetrievalPrecision

from transformers_framework.architectures.modeling_outputs import EmbeddingOutput
from transformers_framework.interfaces.adaptation import retrieval_adaptation
from transformers_framework.interfaces.logging import (
    LOSS,
    RETRIEVAL_ACCURACY,
    RETRIEVAL_F1,
    RETRIEVAL_HR,
    RETRIEVAL_LOSS,
    RETRIEVAL_MAP,
    RETRIEVAL_MRR,
    RETRIEVAL_PRECISION,
)
from transformers_framework.interfaces.step import RetrievalStepOutput
from transformers_framework.pipelines.pipeline import Pipeline
from transformers_framework.processing.postprocessors import retrieval_processor
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_retrieval_arguments
from transformers_framework.utilities.losses import contrastive_loss, cosine_similarity_loss, online_contrastive_loss
from transformers_framework.utilities.similarity import dot_product_similarity
from transformers_framework.utilities.torch import logits_to_binary_predictions


LOSS_FN_MAP = dict(
    cosine_similarity_loss=cosine_similarity_loss,
    contrastive_loss=contrastive_loss,
    online_contrastive_loss=online_contrastive_loss,
)

SCORES_FN_MAP = dict(
    cosine_similarity=cosine_similarity,
    dot_product_similarity=dot_product_similarity,
)


class RetrievalPipeline(Pipeline):
    r"""
    This class will be probably expanded an divided in many subclasses based on the retrieval training type.
    Right now it is designed to process a batch containing only queries and documents.
    """

    POST_FORWARD_ADAPTER = retrieval_adaptation

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        assert self.hyperparameters.loss_fn in LOSS_FN_MAP
        assert self.hyperparameters.scores_fn in SCORES_FN_MAP
        assert self.hyperparameters.k is not None
        assert self.hyperparameters.k == 2  # till we do not implement triple losses and scores

        # setting loss function for every step
        self.loss_function = LOSS_FN_MAP[self.hyperparameters.loss_fn]
        self.scores_fn = SCORES_FN_MAP[self.hyperparameters.scores_fn]

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
        self.valid_p5 = RetrievalPrecision(top_k=5, **metrics_kwargs)
        self.valid_p100 = RetrievalPrecision(top_k=100, **metrics_kwargs)
        self.valid_hr5 = RetrievalHitRate(top_k=5, **metrics_kwargs)
        self.valid_hr100 = RetrievalHitRate(top_k=100, **metrics_kwargs)

        # test metrics
        self.test_acc = BinaryAccuracy(ignore_index=IGNORE_IDX)
        self.test_f1 = BinaryF1Score(ignore_index=IGNORE_IDX)
        self.test_map = RetrievalMAP(**metrics_kwargs)
        self.test_mrr = RetrievalMRR(**metrics_kwargs)
        self.test_p1 = RetrievalPrecision(top_k=1, **metrics_kwargs)
        self.test_p5 = RetrievalPrecision(top_k=5, **metrics_kwargs)
        self.test_p100 = RetrievalPrecision(top_k=100, **metrics_kwargs)
        self.test_hr5 = RetrievalHitRate(top_k=5, **metrics_kwargs)
        self.test_hr100 = RetrievalHitRate(top_k=100, **metrics_kwargs)

        # set model names to reduce manually because it depends on self.hyperparameters.k
        self.MODEL_INPUT_NAMES_TO_REDUCE = [
            (f'input_ids_{i}', f'attention_mask_{i}', f'token_type_ids_{i}')
            for i in range(self.hyperparameters.k)
        ]

    def synchronize_tensor(self, _tensor: Tensor, sync_grads: bool = True) -> Tensor:
        r""" Synchronize tensor from all devices by concatenating along `batch` dimension. """
        new_batch_size = self.trainer.world_size * _tensor.shape[0]
        _tensor = self.all_gather(_tensor, sync_grads=sync_grads).view(new_batch_size, *_tensor.shape[1:])
        return _tensor

    def step(self, batch: Dict) -> RetrievalStepOutput:
        r""" Forward step is shared between all train/val/test steps. """

        embeddings: List[EmbeddingOutput] = [
            self.forward(**minibatch)
            for minibatch in (
                {k[:-2]: v for k, v in batch.items() if k.endswith(f'_{i}')}
                for i in range(self.hyperparameters.k)
            )
        ]

        # extract just last hidden state over CLS from every output object
        embeddings = [embedding.last_hidden_state[:, 0, :] for embedding in embeddings]

        # check all have the same shape
        assert all(embedding.shape == embeddings[0].shape for embedding in embeddings), (
            f"embeddings must have the same shape, found "
            f"{(embedding.shape for embedding in embeddings)} instead"
        )

        # compute scores before synchronizing between devices to avoid storing
        # metrics multiple times for each pair of embeddings
        scores = self.scores_fn(*embeddings)
        preds = logits_to_binary_predictions(scores)

        # computing the loss
        if self.hyperparameters.sync_devices:
            synced_embeddings = [
                self.synchronize_tensor(embedding, sync_grads=True) for embedding in embeddings
            ]
            synced_labels = self.synchronize_tensor(batch['retrieval_labels'], sync_grads=False)
        else:
            synced_embeddings = embeddings
            synced_labels = batch['retrieval_labels']
        
        loss = self.loss_function(*synced_embeddings, labels=synced_labels)

        return RetrievalStepOutput(
            loss=loss,
            retrieval_loss=loss,
            retrieval_index=batch['index'],
            retrieval_scores=scores,
            retrieval_predictions=preds,
            retrieval_labels=batch['retrieval_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        train_acc = self.train_acc(step_output.retrieval_predictions, step_output.retrieval_labels)
        train_f1 = self.train_f1(step_output.retrieval_predictions, step_output.retrieval_labels)

        self.log(LOSS, step_output.loss)
        self.log(RETRIEVAL_LOSS, step_output.retrieval_loss)
        self.log(RETRIEVAL_ACCURACY, train_acc)
        self.log(RETRIEVAL_F1, train_f1)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output = self.step(batch)

        # AS2 metrics should only be computed globally
        self.validation_step_update_metrics(step_output)

        # logging
        val_acc = self.valid_acc(step_output.retrieval_predictions, step_output.retrieval_labels)
        val_f1 = self.valid_f1(step_output.retrieval_predictions, step_output.retrieval_labels)

        self.log(LOSS, step_output.loss)
        self.log(RETRIEVAL_LOSS, step_output.retrieval_loss)
        self.log(RETRIEVAL_ACCURACY, val_acc)
        self.log(RETRIEVAL_F1, val_f1)

    def validation_step_update_metrics(self, step_output: RetrievalStepOutput):
        r""" Update metrics for answer selection. """
        kwargs = dict(
            preds=step_output.retrieval_scores,
            target=step_output.retrieval_labels,
            indexes=step_output.retrieval_index,
        )
        self.valid_map.update(**kwargs)
        self.valid_mrr.update(**kwargs)
        self.valid_p1.update(**kwargs)
        self.valid_p5.update(**kwargs)
        self.valid_p100.update(**kwargs)
        self.valid_hr5.update(**kwargs)
        self.valid_hr100.update(**kwargs)

    def test_step(self, batch, *args):
        r""" Compute predictions and log retrieval results. """
        step_output = self.step(batch)

        # AS2 metrics should only be computed globally
        self.test_step_update_metrics(step_output)

        # logging
        test_acc = self.test_acc(step_output.retrieval_predictions, step_output.retrieval_labels)
        test_f1 = self.test_f1(step_output.retrieval_predictions, step_output.retrieval_labels)

        self.log(LOSS, step_output.loss)
        self.log(RETRIEVAL_LOSS, step_output.retrieval_loss)
        self.log(RETRIEVAL_ACCURACY, test_acc)
        self.log(RETRIEVAL_F1, test_f1)

    def test_step_update_metrics(self, step_output: RetrievalStepOutput):
        r""" Update metrics for answer selection. """
        kwargs = dict(
            preds=step_output.retrieval_scores,
            target=step_output.retrieval_labels,
            indexes=step_output.retrieval_index,
        )
        self.test_map.update(**kwargs)
        self.test_mrr.update(**kwargs)
        self.test_p1.update(**kwargs)
        self.test_p5.update(**kwargs)
        self.test_p100.update(**kwargs)
        self.test_hr5.update(**kwargs)
        self.test_hr100.update(**kwargs)

    def on_validation_epoch_end(self):
        r""" Just log metrics. """
        super().on_validation_epoch_end()
        self.log(RETRIEVAL_MAP, self.valid_map.compute())
        self.log(RETRIEVAL_MRR, self.valid_mrr.compute())
        self.log(RETRIEVAL_PRECISION(1), self.valid_p1.compute())
        self.log(RETRIEVAL_PRECISION(5), self.valid_p5.compute())
        self.log(RETRIEVAL_PRECISION(100), self.valid_p100.compute())
        self.log(RETRIEVAL_HR(5), self.valid_hr5.compute())
        self.log(RETRIEVAL_HR(100), self.valid_hr100.compute())

    def on_test_epoch_end(self):
        r""" Just log metrics. """
        super().on_test_epoch_end()
        self.log(RETRIEVAL_MAP, self.test_map.compute())
        self.log(RETRIEVAL_MRR, self.test_mrr.compute())
        self.log(RETRIEVAL_PRECISION(1), self.test_p1.compute())
        self.log(RETRIEVAL_PRECISION(5), self.test_p5.compute())
        self.log(RETRIEVAL_PRECISION(100), self.test_p100.compute())
        self.log(RETRIEVAL_HR(5), self.test_hr5.compute())
        self.log(RETRIEVAL_HR(100), self.test_hr100.compute())

    def predict_step(self, batch, *args):
        r""" Predict and return scores. """
        return self.step(batch).retrieval_scores

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Sample should contain question, answer and optionally other columns that will be encoded together. """
        return retrieval_processor(
            sample=sample,
            input_columns=self.hyperparameters.input_columns,
            index_column=self.hyperparameters.index_column,
            label_column=self.hyperparameters.label_column,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            k=self.hyperparameters.k,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--input_columns', type=str, nargs=2, required=True)
        parser.add_argument('--label_column', type=str, required=True)
        parser.add_argument('--index_column', type=str, required=False)
        parser.add_argument('--sync_devices', action="store_true")
        parser.add_argument(
            '--loss_fn', type=str, default="contrastive_loss", required=False, choices=LOSS_FN_MAP.keys()
        )
        parser.add_argument(
            '--scores_fn', type=str, default="cosine_similarity", required=False, choices=SCORES_FN_MAP.keys()
        )
        parser.add_argument(
            '-k', type=int, default=2, required=False, help="The number of inputs to encoder separately"
        )
        add_retrieval_arguments(parser)
