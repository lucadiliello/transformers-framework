import math
from argparse import Namespace

import torch

from transformers_framework.schedulers.scheduler import Scheduler
from transformers_framework.utilities.arguments import FlexibleArgumentParser, float_non_negative, int_non_negative


class PolynomialDecayScheduler(Scheduler):
    r""" Decays the learning rate of each parameter group using a polynomial function
    in the given total steps. When last_epoch=-1, sets initial lr as lr.

    Args:
        hyperparameters: (:class:`~argparse.Namespace`):
            Collection of training hyperparameters.
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_training_steps (:obj:`int`):
            The total number of training steps.

    Args through CLI:
        decay_power (:obj:`int`):
            The power for the decay phase.
        num_warmup_steps  (:obj:`int`):
            The number of warmup steps.
    """

    def lr_lambda(self, current_step: int) -> int:
        r""" Compute lambda that is going to scale the learning rate. """

        if current_step < self.hyperparameters.num_warmup_steps:
            decay_factor = current_step / self.hyperparameters.num_warmup_steps
        else:
            adjusted_total_steps = (self.num_training_steps - self.hyperparameters.num_warmup_steps)
            adjusted_current_step = (current_step - self.hyperparameters.num_warmup_steps)

            decay_factor = (
                1.0 - adjusted_current_step / adjusted_total_steps
            ) / (
                1.0 - (adjusted_current_step - 1) / adjusted_total_steps
            )
    
            decay_factor = decay_factor ** self.hyperparameters.decay_power

        return decay_factor

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--decay_power', type=float, default=1.0, required=False, help="Power of polynomial decay")
        parser.add_argument(
            '--num_warmup_steps', type=int_non_negative, default=0, required=False, help="Number of linear warmup steps"
        )


# Copyright 2020 The Google Research Authors.
# Converted to PyTorch by Luca Di liello
class PolynomialLayerwiseDecayScheduler(Scheduler):
    r"""
    Create a polynomially decreasing scheduler.
    Conversion of `https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay`.

    If `layerwise_lr_decay_power` is different from 1.0, the learning rate over each group will
    be multiplied by a factor `f = layerwise_lr_decay_power^(max(depth) - depth)`. `depth` is a
    key that should be defined in every group of parameters, along with the usual `weight_decay`.

    Args:
        hyperparameters: (:class:`~argparse.Namespace`):
            Collection of training hyperparameters.
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_training_steps (:obj:`int`):
            The total number of training steps.

    Args through CLI:
        decay_power (:obj:`int`):
            The power for the decay phase.
        num_warmup_steps  (:obj:`int`):
            The number of warmup steps.
        end_learning_rate (:obj:`float`, `optional`, defaults to 0.0001):
            target learning rate in the last step/epoch.
        lr_decay_power (:obj:`float`, `optional`, defaults to 1.0):
            learning rate decay base for polynomial decay.
        layerwise_lr_decay_power (:obj:`float`, `optional`, defaults to 1.0):
            learning rate decay base for layerwise decay.
        scheduler_cycle (:obj:`bool`, `optional`, defaults to False):
            whether to extend decay steps when global step is higher.
    """

    def __init__(self, hyperparameters: Namespace, optimizer: torch.optim.Optimizer, num_training_steps: int):
        super().__init__(hyperparameters, optimizer, num_training_steps)

        # retrieve depth for each params group
        self.depths = [group['depth'] for group in optimizer.param_groups]

    def _layerwise_decay(self, lrs):
        r""" Have lower learning rates for layers closer to the input.
        Requires that groups passed to the Optimizer are already sorted from the
        closer to the input to the closer the output. """

        return [
            lr * (self.hyperparameters.layerwise_lr_decay_power ** (max(self.depths) - depth))
            for lr, depth in zip(lrs, self.depths)
        ]

    def get_lr(self):

        decay_steps = self.num_training_steps
        current_step = max(0, self.last_epoch)

        # if scheduler_cycle, extend decay_steps if larger than current_step
        if self.hyperparameters.scheduler_cycle:
            decay_steps = decay_steps * math.ceil(current_step / decay_steps)
        else:
            current_step = min(current_step, decay_steps)

        lrs = [
            (
                (base_lr - self.hyperparameters.end_learning_rate) * (
                    1 - current_step / decay_steps
                ) ** (self.hyperparameters.lr_decay_power) + self.hyperparameters.end_learning_rate
            ) * (
                current_step / self.hyperparameters.num_warmup_steps
                if self.hyperparameters.num_warmup_steps > 0 else 1.0
            ) for base_lr in self.base_lrs
        ]

        lrs = self._layerwise_decay(lrs)
        return lrs

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
        super().add_argparse_args(parser)
        parser.add_argument('--num_warmup_steps', type=int_non_negative, default=1000)
        parser.add_argument('--end_learning_rate', type=float_non_negative, default=0.0001)
        parser.add_argument('--lr_decay_power', type=float, default=1.0)
        parser.add_argument('--layerwise_lr_decay_power', type=float_non_negative, default=1.0)
        parser.add_argument('--scheduler_cycle', action='store_true')
