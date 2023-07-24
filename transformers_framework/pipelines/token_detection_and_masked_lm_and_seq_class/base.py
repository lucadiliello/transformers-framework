from typing import Any, Dict, Union

import torch
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from transformers_framework.architectures.modeling_outputs import MaskedLMOutput, TokenDetectionAndSeqClassOutput
from transformers_framework.interfaces.adaptation import masked_lm_adaptation
from transformers_framework.interfaces.logging import (  # MASKED_LM_PERPLEXITY,
    LOSS,
    MASKED_LM_ACCURACY,
    MASKED_LM_LOSS,
    SEQ_CLASS_ACCURACY,
    SEQ_CLASS_F1,
    SEQ_CLASS_LOSS,
    TOKEN_DETECTION_ACCURACY,
    TOKEN_DETECTION_F1,
    TOKEN_DETECTION_LOSS,
)
from transformers_framework.interfaces.step import MaskedLMAndTokenDetectionAndSeqClassStepOutput, SeqClassStepOutput
# from transformers_framework.metrics.perplexity import Perplexity
from transformers_framework.pipelines.pipeline.pipeline import ExtendedPipeline
from transformers_framework.processing.postprocessors import masked_lm_and_seq_class_processor
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import (
    FlexibleArgumentParser,
    add_extended_seq_class_arguments,
    add_masked_lm_and_token_detection_arguments,
    add_masked_lm_arguments,
    add_seq_class_arguments,
)
from transformers_framework.utilities.distributions import sample_from_distribution
from transformers_framework.utilities.logging import rank_zero_warn
from transformers_framework.utilities.torch import combine_losses


class TokenDetectionAndMaskedLMAndSeqClassPipeline(ExtendedPipeline):

    GENERATOR_MODEL_CLASS: PreTrainedModel
    POST_FORWARD_ADAPTER = None
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'original_input_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels')
    ]

    def get_generator_config(self, config: PretrainedConfig):
        r""" Return config for the generator. """
        raise NotImplementedError("This method should be implemented by subclasses")

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.tie_weights()

        # generator metrics
        metric_args = (self.tokenizer.vocab_size, )
        metric_kwargs = dict(average='micro', ignore_index=IGNORE_IDX)

        self.train_mlm_acc = MulticlassAccuracy(*metric_args, **metric_kwargs)
        # self.train_mlm_ppl = Perplexity(ignore_index=IGNORE_IDX)

        # discriminator metrics
        metrics_args = (2, )
        metrics_kwargs = dict(average=None, ignore_index=IGNORE_IDX)

        self.train_discriminator_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.train_discriminator_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

        # classification metrics
        classification_metrics_args = (self.hyperparameters.num_labels, )

        self.train_class_acc = MulticlassAccuracy(*classification_metrics_args, **metrics_kwargs)
        self.train_class_f1 = MulticlassF1Score(*classification_metrics_args, **metrics_kwargs)

        self.valid_class_acc = MulticlassAccuracy(*classification_metrics_args, **metrics_kwargs)
        self.valid_class_f1 = MulticlassF1Score(*classification_metrics_args, **metrics_kwargs)

        self.test_class_acc = MulticlassAccuracy(*classification_metrics_args, **metrics_kwargs)
        self.test_class_f1 = MulticlassF1Score(*classification_metrics_args, **metrics_kwargs)

    def tie_weights(self):
        r""" Do every sort of weight tying here. """

    def requires_extended_tokenizer(self):
        return len(self.hyperparameters.input_columns) > 2 or self.hyperparameters.extended_token_type_ids is not None

    def requires_extended_model(self):
        return self.hyperparameters.k is not None

    def configure_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        # discriminator config
        if self.requires_extended_model():
            CONFIG_CLASS = self.CONFIG_EXTENDED_CLASS
        if self.hyperparameters.k is not None:
            kwargs['k'] = self.hyperparameters.k
        if self.hyperparameters.extended_token_type_ids is not None:
            kwargs['type_vocab_size'] = max(self.hyperparameters.extended_token_type_ids) + 1
        if self.hyperparameters.classification_head_type is not None:
            kwargs['classification_head_type'] = self.hyperparameters.classification_head_type

        config = CONFIG_CLASS.from_pretrained(
            self.hyperparameters.pre_trained_config, num_labels=self.hyperparameters.num_labels, **kwargs
        )

        # generator config
        if self.hyperparameters.pre_trained_generator_config is not None:
            self.generator_config = self.CONFIG_CLASS.from_pretrained(  # this is still the original config class
                self.hyperparameters.pre_trained_generator_config
            )
        elif self.hyperparameters.pre_trained_generator_model is not None:
            rank_zero_warn(
                'Found None `pre_trained_generator_config`, setting equal to `pre_trained_generator_model`'
            )
            self.hyperparameters.pre_trained_generator_config = self.hyperparameters.pre_trained_generator_model
            self.generator_config = self.CONFIG_CLASS.from_pretrained(
                self.hyperparameters.pre_trained_generator_config
            )
        else:
            rank_zero_warn(
                f"Automatically creating generator config of size {self.hyperparameters.generator_size}"
            )
            self.generator_config = self.get_generator_config(config)

        return config

    def configure_model(self, config: PretrainedConfig, **kwargs) -> PreTrainedModel:
        r""" Configure models. """
        if self.requires_extended_model():
            self.MODEL_CLASS = self.MODEL_EXTENDED_CLASS

        generator = self.load_model(
            self.GENERATOR_MODEL_CLASS, self.hyperparameters.pre_trained_generator_model, config=self.generator_config
        )
        model = self.load_model(self.MODEL_CLASS, self.hyperparameters.pre_trained_model, config=config)
        return {'generator': generator, 'model': model}

    def step(self, batch: Dict) -> MaskedLMAndTokenDetectionAndSeqClassStepOutput:
        r""" Forward step on the generator, labels creation and step on the discriminator. """
        original_input_ids = batch.pop('original_input_ids')
        seq_class_labels = batch.pop('seq_class_labels')

        generator_output: MaskedLMOutput = masked_lm_adaptation(self.forward(**batch, model='generator'))

        masked_lm_labels = batch.pop('masked_lm_labels')
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
        batch['seq_class_labels'] = seq_class_labels
        # attention mask and token type ids should be left unchanged

        discriminator_output: TokenDetectionAndSeqClassOutput = self.forward(**batch)

        loss = combine_losses(
            losses=[
                generator_output.masked_lm_loss,
                discriminator_output.token_detection_loss,
                discriminator_output.seq_class_loss,
            ],
            weigths=[
                self.hyperparameters.masked_lm_weight,
                self.hyperparameters.token_detection_weight,
                self.hyperparameters.seq_class_weight,
            ],
        )

        discriminator_predictions = (discriminator_output.token_detection_logits > 0.5).to(dtype=torch.int64)
        seq_class_predictions = discriminator_output.seq_class_logits.argmax(dim=-1)

        return MaskedLMAndTokenDetectionAndSeqClassStepOutput(
            loss=loss,
            token_detection_loss=discriminator_output.token_detection_loss,
            token_detection_predictions=discriminator_predictions,
            token_detection_logits=discriminator_output.token_detection_logits,
            token_detection_labels=discriminator_labels,
            seq_class_loss=discriminator_output.seq_class_loss,
            seq_class_predictions=seq_class_predictions,
            seq_class_labels=seq_class_labels,
            masked_lm_loss=generator_output.masked_lm_loss,
            masked_lm_predictions=generator_predictions,
            masked_lm_logits=generator_logits,
            masked_lm_labels=masked_lm_labels,
        )

    def training_step(self, batch, *args):
        r""" Start by masking some tokens. """
        step_output = self.step(batch)

        train_mlm_acc = self.train_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        # train_mlm_ppl = self.train_mlm_ppl(step_output.masked_lm_logits.float(), step_output.masked_lm_labels)

        train_discriminator_acc = self.train_discriminator_acc(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )
        train_discriminator_f1 = self.train_discriminator_f1(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )

        train_class_acc = self.train_class_acc(step_output.seq_class_predictions, step_output.seq_class_labels)
        train_class_f1 = self.train_class_f1(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(MASKED_LM_LOSS, step_output.masked_lm_loss)
        self.log(MASKED_LM_ACCURACY, train_mlm_acc)
        # self.log(MASKED_LM_PERPLEXITY, train_mlm_ppl)
        self.log(TOKEN_DETECTION_LOSS, step_output.token_detection_loss)
        self.log(TOKEN_DETECTION_ACCURACY, train_discriminator_acc)
        self.log(TOKEN_DETECTION_F1, train_discriminator_f1)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, train_class_acc)
        self.log(SEQ_CLASS_F1, train_class_f1)

        return step_output.loss

    def evaluation_step(self, batch):
        r""" Simple sequence classification over the discriminator. """

        # pop unused keys
        batch.pop('original_input_ids')
        batch.pop('masked_lm_labels')
        batch['token_detection_labels'] = None

        discriminator_output: TokenDetectionAndSeqClassOutput = self.forward(**batch)

        return SeqClassStepOutput(
            loss=discriminator_output.seq_class_loss,
            seq_class_loss=discriminator_output.seq_class_loss,
            seq_class_predictions=discriminator_output.seq_class_predictions,
            seq_class_labels=batch['seq_class_labels'],
        )

    def validation_step(self, batch, *args):
        r""" Start by masking some tokens. """
        step_output = self.evaluation_step(batch)

        valid_class_acc = self.valid_class_acc(step_output.seq_class_predictions, step_output.seq_class_labels)
        valid_class_f1 = self.valid_class_f1(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, valid_class_acc)
        self.log(SEQ_CLASS_F1, valid_class_f1)

    def test_step(self, batch, *args):
        r""" Start by masking some tokens. """
        step_output = self.evaluation_step(batch)

        test_class_acc = self.test_class_acc(step_output.seq_class_predictions, step_output.seq_class_labels)
        test_class_f1 = self.test_class_f1(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, test_class_acc)
        self.log(SEQ_CLASS_F1, test_class_f1)

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add denoising objective. """
        return masked_lm_and_seq_class_processor(
            sample=sample,
            input_columns=self.hyperparameters.input_columns,
            label_column=self.hyperparameters.label_column,
            probability=self.hyperparameters.probability,
            probability_masked=self.hyperparameters.probability_masked,
            probability_replaced=self.hyperparameters.probability_replaced,
            probability_unchanged=self.hyperparameters.probability_unchanged,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            whole_word_masking=self.hyperparameters.whole_word_masking,
            training=self.training,
            extended_token_type_ids=self.hyperparameters.extended_token_type_ids,
            k=self.hyperparameters.k,
            return_original_input_ids=True,
            pad_to_k=self.hyperparameters.pad_to_k,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--masked_lm_weight', type=float, default=1.0)
        parser.add_argument('--token_detection_weight', type=float, default=50.0)
        parser.add_argument('--seq_class_weight', type=float, default=1.0)
        parser.add_argument('--input_columns', type=str, nargs='+', required=True)
        parser.add_argument('--label_column', type=str, required=True)
        add_masked_lm_arguments(parser)
        add_seq_class_arguments(parser)
        add_masked_lm_and_token_detection_arguments(parser)
        add_extended_seq_class_arguments(parser)
