from typing import Any, Dict

import torch
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from transformers_framework.architectures.modeling_outputs import TokenDetectionOutput
from transformers_framework.interfaces.adaptation import masked_lm_adaptation, token_detection_adaptation
from transformers_framework.interfaces.logging import (
    LOSS,
    MASKED_LM_ACCURACY,
    MASKED_LM_LOSS,
    MASKED_LM_PERPLEXITY,
    TOKEN_DETECTION_ACCURACY,
    TOKEN_DETECTION_F1,
    TOKEN_DETECTION_LOSS,
)
from transformers_framework.interfaces.step import MaskedLMAndTokenDetectionStepOutput
from transformers_framework.metrics.perplexity import Perplexity
from transformers_framework.pipelines.pipeline.pipeline import Pipeline
from transformers_framework.processing.postprocessors import masked_lm_processor
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import (
    FlexibleArgumentParser,
    add_masked_lm_and_token_detection_arguments,
    add_masked_lm_arguments,
)
from transformers_framework.utilities.distributions import sample_from_distribution
from transformers_framework.utilities.logging import rank_zero_warn
from transformers_framework.utilities.torch import combine_losses


class TokenDetectionAndMaskedLMPipeline(Pipeline):

    GENERATOR_MODEL_CLASS: PreTrainedModel
    POST_FORWARD_ADAPTER = token_detection_adaptation
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'original_input_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels')
    ]

    def get_generator_config(self, config: PretrainedConfig, **kwargs):
        r""" Return config for the generator. """
        raise NotImplementedError("This method should be implemented by subclasses")

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.tie_weights()

        # generator metrics
        metrics_args = (self.tokenizer.vocab_size, )
        metrics_kwargs = dict(average='micro', ignore_index=IGNORE_IDX)

        self.train_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.valid_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.test_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)

        metrics_kwargs = dict(ignore_index=IGNORE_IDX)
        self.train_mlm_ppl = Perplexity(**metrics_kwargs)
        self.valid_mlm_ppl = Perplexity(**metrics_kwargs)
        self.test_mlm_ppl = Perplexity(**metrics_kwargs)

        # discriminator metrics
        discriminator_metrics_kwargs = dict(average=None, ignore_index=IGNORE_IDX, num_classes=2)

        # train metrics
        self.train_discriminator_acc = MulticlassAccuracy(**discriminator_metrics_kwargs)
        self.train_discriminator_f1 = MulticlassF1Score(**discriminator_metrics_kwargs)

        # validation metrics
        self.valid_discriminator_acc = MulticlassAccuracy(**discriminator_metrics_kwargs)
        self.valid_discriminator_f1 = MulticlassF1Score(**discriminator_metrics_kwargs)

        # test metrics
        self.test_discriminator_acc = MulticlassAccuracy(**discriminator_metrics_kwargs)
        self.test_discriminator_f1 = MulticlassF1Score(**discriminator_metrics_kwargs)

    def tie_weights(self):
        r""" Do every sort of weight tying here. """

    def configure_config(self) -> PretrainedConfig:
        kwargs = dict()

        # discriminator config
        config = self.CONFIG_CLASS.from_pretrained(self.hyperparameters.pre_trained_config, **kwargs)

        # generator config
        if self.hyperparameters.pre_trained_generator_config is not None:
            generator_config = self.CONFIG_CLASS.from_pretrained(
                self.hyperparameters.pre_trained_generator_config, **kwargs
            )
        elif self.hyperparameters.pre_trained_generator_model is not None:
            rank_zero_warn(
                'Found None `pre_trained_generator_config`, setting equal to `pre_trained_generator_model`'
            )
            self.hyperparameters.pre_trained_generator_config = self.hyperparameters.pre_trained_generator_model
            generator_config = self.CONFIG_CLASS.from_pretrained(
                self.hyperparameters.pre_trained_generator_config, **kwargs
            )
        else:
            rank_zero_warn(
                f"Automatically creating generator config of size {self.hyperparameters.generator_size}"
            )
            generator_config = self.get_generator_config(config, **kwargs)

        return {'config': config, 'generator_config': generator_config}

    def configure_model(self, config: PretrainedConfig) -> PreTrainedModel:
        r""" Configure models. """
        generator = self.load_model(
            self.GENERATOR_MODEL_CLASS, self.hyperparameters.pre_trained_generator_model, config=self.generator_config
        )
        model = self.load_model(self.MODEL_CLASS, self.hyperparameters.pre_trained_model, config=config)
        return {'generator': generator, 'model': model}

    def step(self, batch: Dict) -> MaskedLMAndTokenDetectionStepOutput:
        r""" Forward step on the generator, labels creation and step on the discriminator. """
        original_input_ids = batch.pop('original_input_ids')
        masked_lm_labels = batch.pop('masked_lm_labels')

        generator_output = masked_lm_adaptation(self.generator(**batch, labels=masked_lm_labels))
        is_mlm_applied = (masked_lm_labels != IGNORE_IDX)

        with torch.no_grad():
            # take predictions of the generator where a mask token was placed
            generator_logits = generator_output.masked_lm_logits
            generator_generations = sample_from_distribution(
                generator_logits, sample_function=self.hyperparameters.sample_function
            )
            generator_predictions = generator_logits.argmax(dim=-1)
            new_generated_tokens = generator_generations[is_mlm_applied]

            # replace mask tokens with the ones predicted by the generator
            discriminator_input_ids = original_input_ids.clone()
            discriminator_input_ids[is_mlm_applied] = new_generated_tokens

            # create labels for the discriminator as the mask of the original labels (1 where a token was masked)
            # while at the same time avoiding setting a positive label when prediction where the generator was correct
            discriminator_labels = is_mlm_applied.clone()
            discriminator_labels[is_mlm_applied] = (new_generated_tokens != masked_lm_labels[is_mlm_applied])

            discriminator_labels = discriminator_labels.to(dtype=torch.long)

        batch['input_ids'] = discriminator_input_ids
        batch['token_detection_labels'] = discriminator_labels
        # attention mask and token type ids should be left unchanged

        discriminator_output: TokenDetectionOutput = self.forward(**batch)

        loss = combine_losses(
            losses=[generator_output.masked_lm_loss, discriminator_output.token_detection_loss],
            weigths=[self.hyperparameters.masked_lm_weight, self.hyperparameters.token_detection_weight],
        )

        discriminator_predictions = (discriminator_output.token_detection_logits > 0.5).to(dtype=torch.int64)
    
        return MaskedLMAndTokenDetectionStepOutput(
            loss=loss,
            token_detection_loss=discriminator_output.token_detection_loss,
            token_detection_logits=discriminator_output.token_detection_logits,
            token_detection_predictions=discriminator_predictions,
            token_detection_labels=discriminator_labels,
            masked_lm_loss=generator_output.masked_lm_loss,
            masked_lm_predictions=generator_predictions,
            masked_lm_logits=generator_logits,
            masked_lm_labels=masked_lm_labels,
        )

    def training_step(self, batch, *args):
        r""" Start by masking some tokens. """
        step_output = self.step(batch)

        train_mlm_acc = self.train_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        train_mlm_ppl = self.train_mlm_ppl(step_output.masked_lm_logits.float(), step_output.masked_lm_labels)

        train_discriminator_acc = self.train_discriminator_acc(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )
        train_discriminator_f1 = self.train_discriminator_f1(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )

        self.log(LOSS, step_output.loss)
        self.log(MASKED_LM_LOSS, step_output.masked_lm_loss)
        self.log(MASKED_LM_ACCURACY, train_mlm_acc)
        self.log(MASKED_LM_PERPLEXITY, train_mlm_ppl)
        self.log(TOKEN_DETECTION_LOSS, step_output.token_detection_loss)
        self.log(TOKEN_DETECTION_ACCURACY, train_discriminator_acc)
        self.log(TOKEN_DETECTION_F1, train_discriminator_f1)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Start by masking some tokens. """
        step_output = self.step(batch)

        valid_mlm_acc = self.valid_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        valid_mlm_ppl = self.valid_mlm_ppl(step_output.masked_lm_logits.float(), step_output.masked_lm_labels)

        valid_discriminator_acc = self.valid_discriminator_acc(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )
        valid_discriminator_f1 = self.valid_discriminator_f1(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )

        self.log(LOSS, step_output.loss)
        self.log(MASKED_LM_LOSS, step_output.masked_lm_loss)
        self.log(MASKED_LM_ACCURACY, valid_mlm_acc)
        self.log(MASKED_LM_PERPLEXITY, valid_mlm_ppl)
        self.log(TOKEN_DETECTION_LOSS, step_output.token_detection_loss)
        self.log(TOKEN_DETECTION_ACCURACY, valid_discriminator_acc)
        self.log(TOKEN_DETECTION_F1, valid_discriminator_f1)

    def test_step(self, batch, *args):
        r""" Start by masking some tokens. """
        step_output = self.step(batch)

        test_mlm_acc = self.test_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        test_mlm_ppl = self.test_mlm_ppl(step_output.masked_lm_logits.float(), step_output.masked_lm_labels)

        test_discriminator_acc = self.test_discriminator_acc(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )
        test_discriminator_f1 = self.test_discriminator_f1(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )

        self.log(LOSS, step_output.loss)
        self.log(MASKED_LM_LOSS, step_output.masked_lm_loss)
        self.log(MASKED_LM_ACCURACY, test_mlm_acc)
        self.log(MASKED_LM_PERPLEXITY, test_mlm_ppl)
        self.log(TOKEN_DETECTION_LOSS, step_output.token_detection_loss)
        self.log(TOKEN_DETECTION_ACCURACY, test_discriminator_acc)
        self.log(TOKEN_DETECTION_F1, test_discriminator_f1)

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add denoising objective. """
        return masked_lm_processor(
            sample=sample,
            input_columns=self.hyperparameters.input_columns,
            probability=self.hyperparameters.probability,
            probability_masked=self.hyperparameters.probability_masked,
            probability_replaced=self.hyperparameters.probability_replaced,
            probability_unchanged=self.hyperparameters.probability_unchanged,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            whole_word_masking=self.hyperparameters.whole_word_masking,
            return_original_input_ids=True,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--masked_lm_weight', type=float, default=1.0)
        parser.add_argument('--token_detection_weight', type=float, default=50.0)
        parser.add_argument('--input_columns', nargs='+', type=str, required=True)
        add_masked_lm_arguments(parser)
        add_masked_lm_and_token_detection_arguments(parser)
