from transformers_framework.schedulers.constant import ConstantScheduler
from transformers_framework.schedulers.cosine import CosineScheduler
from transformers_framework.schedulers.linear import LinearScheduler
from transformers_framework.schedulers.max_sqrt import MaxSQRTDecayScheduler
from transformers_framework.schedulers.normal_decay import NormalDecayScheduler
from transformers_framework.schedulers.polynomial_decay import (
    PolynomialDecayScheduler,
    PolynomialLayerwiseDecayScheduler,
)


schedulers = dict(
    constant=ConstantScheduler,
    cosine=CosineScheduler,
    polynomial_decay_layerwise=PolynomialLayerwiseDecayScheduler,
    linear_decay=LinearScheduler,
    sqrt_decay=MaxSQRTDecayScheduler,
    polynomial_decay=PolynomialDecayScheduler,
    normal_decay=NormalDecayScheduler,
)
