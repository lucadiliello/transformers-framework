from argparse import ArgumentParser
from math import exp


class NormalDecayScheduler:
    r"""
    Create a schedule with a learning rate that descreses following a normal distribution before and after
    the first `k` steps. The lr function is
    `lr = {x < num_warmup_steps: e^(-(x-num_warmup_steps)*2/num_warmup_steps)^2, e^(-(x-num_warmup_steps)/num_training_steps)^2)`

    """

    def __init__(self, num_warmup_steps: int, num_training_steps: int) -> None:
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps

    def lr_lambda(self, current_step: int) -> int:
        r""" Compute lambda that is going to scale the learning rate. """

        if self.num_warmup_steps is not None and current_step < self.num_warmup_steps:
            exponential = (current_step - self.num_warmup_steps) * 2 / self.num_warmup_steps
        else:
            exponential = (current_step - self.num_warmup_steps) / self.num_training_steps

        decay_factor = exp(-(exponential ** 2))
        return decay_factor


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--warmup', default=None, type=int)
    parser.add_argument('--total', required=True, type=int)
    parser.add_argument('--steps', default=30, type=int)

    args = parser.parse_args()
    scheduler = NormalDecayScheduler(args.warmup, args.total)

    for i in range(0, args.total + 1, args.total // args.steps):
        print(f"Step {i}\t{scheduler.lr_lambda(i)}")
