from math import sqrt

from transformers_framework.schedulers.scheduler import Scheduler
from transformers_framework.utilities.arguments import FlexibleArgumentParser, int_positive


class MaxSQRTDecayScheduler(Scheduler):
    r"""
    Create a schedule with a learning rate that descreses polynomially after the first `k` steps.
    The lr function is `lr = 1 / sqrt(max(num_constant_steps, step))`

    Args:
        hyperparameters: (:class:`~argparse.Namespace`):
            Collection of training hyperparameters.
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_training_steps (:obj:`int`):
            The total number of training steps.

    Args through CLI:
        num_constant_steps (:obj:`int`):
            The number of steps for the contant phase.
        num_warmup_steps (:obj:`int`):
            The number of steps for the linear increasing phase.
    """

    def lr_lambda(self, current_step: int) -> int:
        r""" Compute lambda that is going to scale the learning rate. """

        use_warmup = self.hyperparameters.num_warmup_steps is not None
        num_steps = (
            self.hyperparameters.num_warmup_steps
            if self.hyperparameters.num_warmup_steps is not None
            else self.hyperparameters.num_constant_steps
        )

        if use_warmup and current_step < num_steps:
            decay_factor = current_step / num_steps
        else:
            decay_factor = 1 / sqrt(max(num_steps, current_step) / num_steps)

        return decay_factor

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
        super().add_argparse_args(parser)
        group_ex = parser.add_mutually_exclusive_group(required=True)
        group_ex.add_argument('--num_constant_steps', type=int_positive, default=None)
        group_ex.add_argument('--num_warmup_steps', type=int_positive, default=None)
