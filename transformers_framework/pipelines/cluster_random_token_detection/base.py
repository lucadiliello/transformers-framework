from typing import Any, Dict

import torch

from transformers_framework.callbacks.save_additional_data import SaveDataCallback
from transformers_framework.interfaces.logging import (
    LOSS,
    TOKEN_DETECTION_ACCURACY,
    TOKEN_DETECTION_F1,
    TOKEN_DETECTION_LOSS,
    TOKEN_DETECTION_PERPLEXITY,
)
from transformers_framework.pipelines.random_token_detection.base import RandomTokenDetectionPipeline
from transformers_framework.processing.postprocessors import clustered_random_token_detection_processor
from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.readers import read_clusters


class ClusterRandomTokenDetectionPipeline(RandomTokenDetectionPipeline):
    r"""
    A model that use RTS loss where the probability of swapping each token is weighted
    by the experience of previous similar switchings.
    """

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self.register_buffer(
            'token_to_cluster_map', read_clusters(self.hyperparameters.clusters_filename), persistent=True,
        )  # token -> clusters
        number_of_clusters = self.token_to_cluster_map.max() + 1

        self.register_buffer(
            'counts', torch.ones(number_of_clusters, number_of_clusters, dtype=torch.int64), persistent=True,
        )  # clusters -> clusters

        self.update_references()

    def configure_callbacks(self):
        return SaveDataCallback(self.hyperparameters, 'counts')

    def update_references(self):
        self.token_to_cluster_map_numpy = self.token_to_cluster_map.cpu().detach().numpy()
        self.counts_numpy = self.counts.cpu().detach().numpy()

    def update_count_vector(
        self,
        originals: torch.Tensor = None,
        tampereds: torch.Tensor = None,
        predictions: torch.Tensor = None,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        r"""
        Update the vector of counts based on new predicitions.

        Args:
            originals:
                original ids of shape (batch_size, max_sequence_len)
            tampereds:
                modified ids of shape (batch_size, max_sequence_len)
            attention_mask:
                attention mask to update only on relevant positions of shape (batch_size, max_sequence_len)
            predictions:
                predictions for each modified ids of shape (batch_size, max_sequence_len)
            labels:
                gold labels of rts of shape (batch_size, max_sequence_len)

        Example:
            >>> attention_mask = torch.tensor([1, 1, 1, 1, 1, 1])
            >>> originals = torch.tensor([2, 3, 56, 1, 2, 23])
            >>> tampereds = torch.tensor([2, 33, 76, 1, 2, 28])
            >>> predictions = torch.tensor([0, 1, 0, 1, 1, 0])
            >>> labels = torch.tensor([0, 1, 1, 0, 0, 0])
            >>> updates = (predictions != labels) * 2 - 1
            torch.tensor([-1, -1, 1, 1, 1, -1])
        """

        indexes = (attention_mask == 1) if attention_mask is not None else torch.full_like(originals, fill_value=True)
        if self.hyperparameters.update_only_on_predictions:
            indexes = indexes & (originals != tampereds)    # select positions where something changed

        originals = originals[indexes]
        predictions = predictions[indexes]
        tampereds = tampereds[indexes]
        labels = labels[indexes]

        originals_clusters = self.token_to_cluster_map[originals]
        tampereds_clustures = self.token_to_cluster_map[tampereds]

        updates_matrix = torch.zeros_like(self.counts)
        updates_matrix[originals_clusters, tampereds_clustures] += (predictions != labels) * 2 - 1

        # gather changes from other processes
        updates_matrix = self.all_gather(updates_matrix).sum(dim=0)

        # works in distributed because it is registered as buffer
        self.counts += updates_matrix

        self.update_references()

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """

        original_input_ids = batch.pop('original_input_ids')
        step_output = self.step(batch)

        self.update_count_vector(
            originals=original_input_ids,
            tampereds=batch['input_ids'],
            predictions=step_output.token_detection_predictions,
            labels=step_output.token_detection_labels,
            attention_mask=batch.get('attention_mask', None),
        )

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
        batch.pop('original_input_ids')
        return super().validation_step(batch, *args)

    def test_step(self, batch, *args):
        batch.pop('original_input_ids')
        return super().test_step(batch, *args)

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add denoising objective. """
        return clustered_random_token_detection_processor(
            sample=sample,
            input_columns=self.hyperparameters.input_columns,
            probability=self.hyperparameters.probability,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            whole_word_detection=self.hyperparameters.whole_word_detection,
            token_to_cluster_map=self.token_to_cluster_map_numpy,
            counts=self.counts_numpy,
            beta=self.hyperparameters.beta,
            list_forbitten_replacements=self.tokenizer.all_special_ids,
        )

    @classmethod
    def add_argparse_args(self, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--beta', default=2.0, required=False, type=float)
        parser.add_argument('--update_only_on_predictions', action="store_true")
        parser.add_argument('--clusters_filename', type=str, required=True)
        SaveDataCallback.add_argparse_args(parser)  # add arguments from callback
