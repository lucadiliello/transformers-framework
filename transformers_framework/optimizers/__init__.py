from transformers_framework.optimizers.adafactor import AdafactorOptimizer
from transformers_framework.optimizers.adamw import AdamWOptimizer
from transformers_framework.optimizers.bistandbytes import Adam8bitOptimizer, AdamW8bitOptimizer
from transformers_framework.optimizers.fuse_adam import FuseAdamOptimizer
from transformers_framework.optimizers.tf_adamw import TFAdamWOptimizer


optimizers = dict(
    adafactor=AdafactorOptimizer,
    fuse_adam=FuseAdamOptimizer,
    adamw=AdamWOptimizer,
    tf_adamw=TFAdamWOptimizer,
    adam8bit=Adam8bitOptimizer,
    adamw8bit=AdamW8bitOptimizer,
)
