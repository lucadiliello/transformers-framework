from math import exp

from transformers_framework.schedulers.scheduler import Scheduler
from transformers_framework.utilities.arguments import FlexibleArgumentParser, int_positive


class NormalDecayScheduler(Scheduler):
    r"""
    Create a schedule with a learning rate that descreses following a normal distribution before and after
    the first `k` steps. The lr function is
    `lr = {
        x < num_warmup_steps:
        e^(-(x-num_warmup_steps)*2/num_warmup_steps)^2,
        e^(-(x-num_warmup_steps)/num_training_steps)^2),
    }`

    Args:
        hyperparameters: (:class:`~argparse.Namespace`):
            Collection of training hyperparameters.
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_training_steps (:obj:`int`):
            The total number of training steps.

    Args through CLI:
        num_warmup_steps (:obj:`int`):
            Whether to use constant or warmup for the first `num_warmup_steps`
    """

    def lr_lambda(self, current_step: int) -> int:
        r""" Compute lambda that is going to scale the learning rate. """

        if self.hyperparameters.num_warmup_steps is not None and current_step < self.hyperparameters.num_warmup_steps:
            exponential = (
                current_step - self.hyperparameters.num_warmup_steps
            ) * 2 / self.hyperparameters.num_warmup_steps
        else:
            exponential = (current_step - self.hyperparameters.num_warmup_steps) / self.num_training_steps

        decay_factor = exp(-(exponential ** 2))
        return decay_factor

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
        super().add_argparse_args(parser)
        parser.add_argument('--num_warmup_steps', type=int_positive, default=None)
