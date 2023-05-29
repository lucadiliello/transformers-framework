
from transformers_framework.optimizers import optimizers
from transformers_framework.optimizers.optimizer import Optimizer
from transformers_framework.schedulers import schedulers
from transformers_framework.schedulers.scheduler import Scheduler
from transformers_framework.utilities.arguments import FlexibleArgumentParser


class OptimizersMixin:

    def num_training_steps(self) -> int:
        r""" Total training steps inferred from datasets length, number of nodes and devices. """
        return (
            self.hyperparameters.max_steps
            if self.hyperparameters.max_steps is not None and self.hyperparameters.max_steps > 0
            else self.trainer.estimated_stepping_batches
        )

    def get_optimizer(self) -> Optimizer:
        r""" Get optimizer as defined by hyperparameters. """
        optim_class = optimizers[self.hyperparameters.optimizer]
        return optim_class(self.hyperparameters, self.named_parameters())

    def get_scheduler(self, optimizer) -> Scheduler:
        r""" Get scheduler as defined by hyperparameters. """
        sched_class = schedulers[self.hyperparameters.scheduler]
        return sched_class(self.hyperparameters, optimizer, self.num_training_steps())

    def configure_optimizers(self):
        r""" Instantiate an optimizer on the parameters of self.model.
        A linear scheduler is also instantiated to manage the learning rate. """

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler':
                {
                    'scheduler': scheduler,  # The LR schduler
                    'interval': self.hyperparameters.scheduler_interval,  # The unit of the scheduler's step size
                    'frequency': self.hyperparameters.scheduler_frequency,  # The frequency of the scheduler
                }
        }

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        parser.add_argument('--optimizer', type=str, default='adamw', choices=optimizers.keys())
        parser.add_argument('--scheduler', type=str, default='linear_decay', choices=schedulers.keys())
        parser.add_argument('--scheduler_interval', type=str, default='step', choices=('step', 'epoch'))
        parser.add_argument('--scheduler_frequency', type=int, default=1)

        # retrieving classes with temporary parsered arguments
        tmp_params, _ = parser.parse_known_args()

        # get pl_model_class in advance to know which params it needs
        optimizers[tmp_params.optimizer].add_argparse_args(parser)
        schedulers[tmp_params.scheduler].add_argparse_args(parser)
