from transformers_framework.utilities.classes import ExtendedNamespace

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.logging import rank_zero_warn


class Scheduler(_LRScheduler):
    r"""
    Create a scheduler with variable learning rate. This class shoud be subclassed.

    More informations about the default parameters can be found on the documentation of
    `_LRScheduler` in the `torch` project.

    Args:
        hyperparameters: (:class:`~argparse.ExtendedNamespace`):
            Collection of training hyperparameters.
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_training_steps (:obj:`int`):
            The total number of training steps.
    """

    def __init__(self, hyperparameters: ExtendedNamespace, optimizer: Optimizer, num_training_steps: int):
        self.hyperparameters = hyperparameters
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, verbose=False)

    def load_state_dict(self, state_dict):
        r"""Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        if state_dict['num_training_steps'] != self.num_training_steps:
            rank_zero_warn(
                f"Restoring Scheduler state with new num_training_steps={self.num_training_steps} "
                f"while checkpoint had num_training_steps={state_dict['num_training_steps']}."
            )
            del state_dict['num_training_steps']
        self.__dict__.update(state_dict)

    def get_lr(self):
        # avoid computing learning rate for -1
        current_step = max(self.last_epoch, 0)

        if current_step > self.num_training_steps:
            raise ValueError(
                f"`current step` {current_step} greater than maximum number of steps {self.num_training_steps}"
            )

        return [base_lr * self.lr_lambda(current_step) for base_lr in self.base_lrs]

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
