from transformers_framework.schedulers.scheduler import Scheduler
from transformers_framework.utilities.arguments import FlexibleArgumentParser, int_non_negative


class ConstantScheduler(Scheduler):
    r"""
    Create a schedule with a learning rate that keeps a contant value
    but increases linearly for the first `num_warmup_steps`.


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
    """

    def lr_lambda(self, current_step: int) -> int:
        r""" Compute lambda that is going to scale the learning rate. """

        if current_step < self.hyperparameters.num_warmup_steps:
            decay_factor = current_step / self.hyperparameters.num_warmup_steps
        else:
            decay_factor = 1.0

        return decay_factor

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
        super().add_argparse_args(parser)
        parser.add_argument('--num_warmup_steps', type=int_non_negative, default=0)
