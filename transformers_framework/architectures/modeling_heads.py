import torch
from torch import nn


class ClassificationHead(nn.Module):
    r""" Head for sequence-level classification tasks. """

    def __init__(self, config, hidden_size: int = None, num_labels: int = None):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
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
