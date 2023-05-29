from transformers_framework.optimizers.adafactor import AdafactorOptimizer
from transformers_framework.optimizers.adamw import AdamWOptimizer
from transformers_framework.optimizers.fuse_adam import FuseAdamOptimizer
from transformers_framework.optimizers.hybrid_adam import CPUAdamColossalAIOptimizer, HybridAdamColossalAIOptimizer
from transformers_framework.optimizers.tf_adamw import TFAdamWOptimizer


optimizers = dict(
    adafactor=AdafactorOptimizer,
    fuse_adam=FuseAdamOptimizer,
    adamw=AdamWOptimizer,
    tf_adamw=TFAdamWOptimizer,
    colossalai_cpu_adam=CPUAdamColossalAIOptimizer,
    colossalai_hybrid_adam=HybridAdamColossalAIOptimizer,
)
