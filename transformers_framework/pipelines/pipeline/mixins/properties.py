from inspect import signature

from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.datamodules import TrainerFn


class PropertiesMixin:

    @property
    def is_training(self):
        return self.trainer.state.fn == TrainerFn.FITTING

    @property
    def is_validation(self):
        return self.trainer.state.fn == TrainerFn.VALIDATING

    @property
    def is_testing(self):
        return self.trainer.state.fn == TrainerFn.TESTING

    @property
    def requires_attention_mask(self):
        return 'attention_mask' in signature(self.model.forward).parameters

    @property
    def requires_decoder_attention_mask(self):
        return 'decoder_attention_mask' in signature(self.model.forward).parameters

    @property
    def requires_token_type_ids(self):
        return 'token_type_ids' in signature(self.model.forward).parameters

    @property
    def requires_decoder_token_type_ids(self):
        return 'decoder_token_type_ids' in signature(self.model.forward).parameters

    def requires_parameter(self, parameter_name):
        return parameter_name in signature(self.model.forward).parameters

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        ...
