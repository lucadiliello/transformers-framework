import math

from transformers_framework.schedulers.scheduler import Scheduler
from transformers_framework.utilities.arguments import FlexibleArgumentParser, int_non_negative


class CosineScheduler(Scheduler):
    r"""
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        hyperparameters: (:class:`~argparse.Namespace`):
            Collection of training hyperparameters.
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_training_steps (:obj:`int`):
            The total number of training steps.

    Args through CLI:
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
    """

    def lr_lambda(self, current_step):
        r""" Compute lambda that is going to scale the learning rate. """

        if current_step < self.hyperparameters.num_warmup_steps:
            decay_factor = current_step / self.hyperparameters.num_warmup_steps
        else:
            adjusted_total_steps = self.num_training_steps - self.hyperparameters.num_warmup_steps
            adjusted_current_step = current_step - self.hyperparameters.num_warmup_steps

            progress = adjusted_current_step / adjusted_total_steps
            decay_factor = max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.hyperparameters.num_cycles) * 2.0 * progress))
            )

        return decay_factor

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
        super().add_argparse_args(parser)
        parser.add_argument('--num_warmup_steps', type=int_non_negative, default=0)
        parser.add_argument('--num_cycles', type=float, default=1.0)
