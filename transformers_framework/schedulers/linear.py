from transformers_framework.schedulers.scheduler import Scheduler
from transformers_framework.utilities.arguments import FlexibleArgumentParser, int_non_negative


class LinearScheduler(Scheduler):
    r"""
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

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

        if self.hyperparameters.num_warmup_steps > 0 and current_step < self.hyperparameters.num_warmup_steps:
            decay_factor = current_step / self.hyperparameters.num_warmup_steps
        else:
            adjusted_total_steps = max(self.num_training_steps - self.hyperparameters.num_warmup_steps, 1)
            adjusted_current_step = current_step - self.hyperparameters.num_warmup_steps

            decay_factor = (adjusted_total_steps - adjusted_current_step) / adjusted_total_steps

        return decay_factor

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
        super().add_argparse_args(parser)
        parser.add_argument('--num_warmup_steps', type=int_non_negative, default=0)
