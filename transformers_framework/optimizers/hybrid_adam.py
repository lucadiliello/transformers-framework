from argparse import Namespace
from typing import Generator

from lightning_utilities.core.imports import RequirementCache

from transformers_framework.optimizers.optimizer import Optimizer
from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.strategy import check_strategy


_COLOSSALAI_AVAILABLE = RequirementCache("colossalai")
if _COLOSSALAI_AVAILABLE:
    from colossalai.nn.optimizer import CPUAdam, HybridAdam
else:
    CPUAdam, HybridAdam = object, object


class ColossalAIOptimizer(Optimizer):

    def __init__(self, hyperparameters: Namespace, named_parameters: Generator):
        r""" First hyperparameters argument to SuperOptimizer, other args for Adafactor. """

        assert _COLOSSALAI_AVAILABLE, (
            "Hybrid Adam optimizer needs `colossalai` to be installed. "
            "Please run `pip install -r requirements/extras.txt`"
        )

        if not check_strategy(hyperparameters.strategy, "colossalai"):
            raise ValueError("`ColossalAIOptimizer` should be used only with `colossalai` strategy.")

        super().__init__(
            hyperparameters,
            named_parameters,
            lr=hyperparameters.learning_rate,
            bias_correction=True,
            betas=hyperparameters.adam_betas,
            eps=hyperparameters.adam_epsilon,
            weight_decay=hyperparameters.weight_decay,
            adamw_mode=True,
            amsgrad=hyperparameters.adam_amsgrad,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        parser.add_argument('--adam_betas', nargs=2, type=float, default=[0.9, 0.999])
        parser.add_argument('--adam_amsgrad', action='store_true')


class CPUAdamColossalAIOptimizer(ColossalAIOptimizer, CPUAdam):
    pass


class HybridAdamColossalAIOptimizer(ColossalAIOptimizer, HybridAdam):
    pass
