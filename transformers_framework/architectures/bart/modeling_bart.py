from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel, logger, shift_tokens_right

from transformers_framework.architectures.bart.configuration_bart import BartMultiTokenConfig
from transformers_framework.architectures.generation_utils import MultiTokenGenerationMixin
from transformers_framework.architectures.modeling_outputs import SeqToSeqLMOutput
from transformers_framework.utilities import IGNORE_IDX


class BartForMultiTokenConditionalGeneration(BartPretrainedModel, MultiTokenGenerationMixin):

    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartMultiTokenConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, config.vocab_size)))
        self.projection = nn.Linear(config.d_model, config.max_multi_token_predictions * config.d_model)
        self.config = config

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], SeqToSeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the seq2seq LM classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`. If labels have shape (batch_size, sequence_length), they
            are automatically converted to shape (batch_size, sequence_length, labels_per_token).

        Returns: SeqToSeqLMOutput object
        """

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs[0]  # shape (batch_size, sequence_length, hidden_size)

        sequence_output = sequence_output[:, ::self.config.max_multi_token_predictions]
        # shape (batch_size, sequence_length // max_multi_token_predictions, hidden_size)

        sequence_output = self.projection(sequence_output)
        # shape (batch_size, sequence_length // max_multi_token_predictions, hidden_size * max_multi_token_predictions)

        sequence_output = sequence_output.reshape(
            *sequence_output.shape[:2], self.config.max_multi_token_predictions, -1
        )
        # shape (batch_size, sequence_length // max_multi_token_predictions, max_multi_token_predictions, hidden_size)

        lm_logits = self.lm_head(sequence_output) + self.final_logits_bias
        # shape (batch_size, sequence_length // max_multi_token_predictions, max_multi_token_predictions, vocab_size)

        lm_logits = lm_logits.view(lm_logits.shape[0], -1, lm_logits.shape[-1])
        # shape (batch_size, sequence_length, vocab_size)

        masked_lm_loss = None
        if labels is not None:
            lm_logits = lm_logits[:, :labels.shape[-1]]  # make sure to match labels seq length
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_IDX)
            masked_lm_loss = loss_fct(lm_logits.reshape(-1, lm_logits.shape[-1]), labels.view(-1))

        return SeqToSeqLMOutput(
            seq_to_seq_lm_loss=masked_lm_loss,
            seq_to_seq_lm_logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            last_hidden_state=outputs.encoder_last_hidden_state,
            hidden_states=outputs.encoder_hidden_states,
            attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
