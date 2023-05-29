import torch
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from transformers.activations import get_activation
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    ContextPooler,
    DebertaV2Model,
    DebertaV2OnlyMLMHead,
    DebertaV2PreTrainedModel,
    StableDropout,
)

from transformers_framework.architectures.modeling_outputs import (
    MaskedLMAndSeqClassOutput,
    TokenDetectionAndSeqClassOutput,
    TokenDetectionOutput,
)
from transformers_framework.utilities import IGNORE_IDX


# Copied from transformers.models.deberta.modeling_deberta.DebertaForMaskedLM with Deberta->DebertaV2
class DebertaV2ForMaskedLMAndSequenceClassification(DebertaV2PreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)

        # MLM
        self.cls = DebertaV2OnlyMLMHead(config)

        # Class
        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        seq_class_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        masked_lm_loss = None
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_IDX)  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        class_loss = None
        if seq_class_labels is not None:
            loss_fct = CrossEntropyLoss()
            class_loss = loss_fct(logits.view(-1, self.num_labels), seq_class_labels.view(-1))

        return MaskedLMAndSeqClassOutput(
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            seq_class_logits=logits,
            seq_class_loss=class_loss,
            masked_lm_logits=prediction_scores,
            masked_lm_loss=masked_lm_loss,
        )


class DebertaDiscriminatorPredictions(nn.Module):
    r"""Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)
        return logits


class DebertaV2EmbeddingsForPreTraining(nn.Module):
    r""" Construct the embeddings from word, position and token_type embeddings for pretraining. """

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)
        self.word_embeddings_delta = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            with torch.no_grad():
                inputs_embeds = self.word_embeddings(input_ids)
            inputs_embeds += self.word_embeddings_delta(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaV2ForPreTraining(DebertaV2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.add_special_embedding_layer()
        self.discriminator_predictions = DebertaDiscriminatorPredictions(config)
        self.post_init()

    def add_special_embedding_layer(self):
        self.deberta.embeddings = DebertaV2EmbeddingsForPreTraining(self.config)

    def get_state_dict_without_special_embedding_layer(self):
        state_dict = self.state_dict()
        state_dict['deberta.embeddings.word_embeddings.weight'] = (
            state_dict['deberta.embeddings.word_embeddings_delta.weight']
            + state_dict['deberta.embeddings.word_embeddings.weight']
        )
        del state_dict['deberta.embeddings.word_embeddings_delta.weight']
        return state_dict

    def save_pretrained(self, *args, **kwargs):
        r""" Merge special pretraining embeddings and save model. """
        state_dict = self.get_state_dict_without_special_embedding_layer()
        return super().save_pretrained(*args, state_dict=state_dict, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the DeBERTa loss. Input should be a sequence of tokens
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        discriminator_sequence_output = discriminator_hidden_states.last_hidden_state
        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        return TokenDetectionOutput(
            token_detection_loss=loss,
            token_detection_logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class DebertaV2ForPreTrainingAndSequenceClassification(DebertaV2PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.add_special_embedding_layer()
        self.discriminator_predictions = DebertaDiscriminatorPredictions(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.post_init()

    def add_special_embedding_layer(self):
        self.deberta.embeddings = DebertaV2EmbeddingsForPreTraining(self.config)

    def get_state_dict_without_special_embedding_layer(self):
        state_dict = self.state_dict()
        state_dict['deberta.embeddings.word_embeddings.weight'] = (
            state_dict['deberta.embeddings.word_embeddings_delta.weight']
            + state_dict['deberta.embeddings.word_embeddings.weight']
        )
        del state_dict['deberta.embeddings.word_embeddings_delta.weight']
        return state_dict

    def save_pretrained(self, *args, **kwargs):
        r""" Merge special pretraining embeddings and save model. """
        state_dict = self.get_state_dict_without_special_embedding_layer()
        return super().save_pretrained(*args, state_dict=state_dict, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        token_detection_labels=None,
        seq_class_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the DeBERTa loss. Input should be a sequence of tokens
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        discriminator_sequence_output = discriminator_hidden_states.last_hidden_state
        logits = self.discriminator_predictions(discriminator_sequence_output)

        pooled_output = self.pooler(discriminator_sequence_output)
        pooled_output = self.dropout(pooled_output)
        class_logits = self.classifier(pooled_output)

        td_loss = None
        if token_detection_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = token_detection_labels[active_loss]
                td_loss = loss_fct(active_logits, active_labels.float())
            else:
                td_loss = loss_fct(
                    logits.view(-1, discriminator_sequence_output.shape[1]), token_detection_labels.float()
                )

        class_loss = None
        if seq_class_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_IDX)
            class_loss = loss_fct(class_logits.view(-1, self.config.num_labels), seq_class_labels.view(-1))

        return TokenDetectionAndSeqClassOutput(
            last_hidden_state=discriminator_sequence_output,
            seq_class_loss=class_loss,
            seq_class_logits=class_logits,
            token_detection_loss=td_loss,
            token_detection_logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
