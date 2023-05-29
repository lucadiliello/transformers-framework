from typing import Union

from pytorch_lightning.strategies.strategy import Strategy


def check_strategy(strategy: Union[Strategy, str], name: str) -> bool:
    r""" Check if used strategy is `name`. """
    if not strategy:
        return None

    if isinstance(strategy, Strategy):
        strategy = getattr(strategy, "strategy_name", strategy.__class__.__name__)

    return strategy.startswith(name)
