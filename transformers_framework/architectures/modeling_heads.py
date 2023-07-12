import torch
from torch import nn

from transformers_framework.architectures.configuration_utils import ExtendedConfig


class ClassificationHead(nn.Module):
    r""" Head for sequence-level classification tasks. """

    def __init__(self, config, hidden_size: int = None, num_labels: int = None):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, 'classifier_dropout', None) is not None
            else config.hidden_dropout_prob
        )
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        num_labels = num_labels if num_labels is not None else config.num_labels

        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        features = self.dropout(features)
        features = self.dense(features)
        features = torch.tanh(features)
        features = self.dropout(features)
        features = self.out_proj(features)
        return features


class ExtendedClassificationHead(nn.Module):
    r""" Head for multiple sequence-level classification tasks. """

    def __init__(self, config: ExtendedConfig):
        super().__init__()
        self.config = config

        if self.config.classification_head_type in ('IE_1', 'IE_k', 'AE_1'):
            self.classifier = ClassificationHead(
                config, hidden_size=config.hidden_size, num_labels=config.num_labels
            )
        elif self.config.classification_head_type in ('AE_k', ):
            self.classifier = ClassificationHead(
                config, hidden_size=config.hidden_size * 2, num_labels=config.num_labels
            )
        elif self.config.classification_head_type in ('RE_k', ):
            self.classifier = ClassificationHead(
                config, hidden_size=config.hidden_size, num_labels=config.k * config.num_labels
            )
        else:
            raise ValueError(f"Head type {self.config.classification_head_type} not among allowed")

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_state: the last hidden_state of some model with shape (batch_size, max_sequence_length, hidden_size)

        Return:
            the logits of shape (batch_size, k, num_labels)
        """

        seq_length = hidden_state.shape[1]

        if self.config.classification_head_type == 'IE_1':
            # single label classification over the first CLS
            hidden_state = hidden_state[:, 0, :]
            features = self.classifier(hidden_state)  # (batch_size, num_labels)
        
        elif self.config.classification_head_type == 'IE_k':
            # multiple classifications over the first output token of the k sequences
            assert seq_length % (self.config.k + 1) == 0  # make sure it is divisible by k
            seq_length = seq_length // (self.config.k + 1)
            cls_positions = [seq_length * i for i in range(self.config.k + 1)]
            hidden_state = hidden_state[:, cls_positions[1:], :]
            features = self.classifier(hidden_state)  # (batch_size, k, num_labels)

        elif self.config.classification_head_type == 'AE_1':
            # single classification over the first output token of the k sequences
            assert seq_length % (self.config.k + 1) == 0  # make sure it is divisible by k
            seq_length = seq_length // (self.config.k + 1)
            cls_positions = [seq_length * i for i in range(self.config.k + 1)]
            hidden_state = hidden_state[:, cls_positions[1:], :].sum(dim=1)
            features = self.classifier(hidden_state)  # (batch_size, num_labels)

        elif self.config.classification_head_type == 'AE_k':
            # multiple classification on the concatenation of the first CLS with all the others
            hidden_state = torch.cat([
                hidden_state[:, cls_positions[0], :].repeat(1, self.config.k, 1),
                hidden_state[:, cls_positions[1:], :],
            ], dim=-1)
            features = self.classifier(hidden_state)  # (batch_size, k, num_labels)

        elif self.config.classification_head_type == 'RE_k':
            # multiple classification all on the first CLS
            hidden_state = hidden_state[:, 0, :]
            features = self.classifier(hidden_state)  # (batch_size, k * num_labels)
            features = features.view(-1, self.config.k, self.config.num_labels)

        return features
