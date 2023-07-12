from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    ContextPooler,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    StableDropout,
)

from transformers_framework.architectures.deberta_v2.configuration_deberta_v2 import DebertaV2ExtendedConfig
from transformers_framework.architectures.deberta_v2.modeling_deberta_v2 import (
    DebertaDiscriminatorPredictions,
    DebertaV2EmbeddingsForPreTraining,
)
from transformers_framework.architectures.modeling_heads import ExtendedClassificationHead
from transformers_framework.architectures.modeling_outputs import SeqClassOutput, TokenDetectionAndSeqClassOutput
from transformers_framework.utilities import IGNORE_IDX


class DebertaV2ForExtendedSequenceClassification(DebertaV2PreTrainedModel):

    def __init__(self, config: DebertaV2ExtendedConfig):
        assert isinstance(config, DebertaV2ExtendedConfig), (
            "ExtendedConfig is required for ExtendedSequenceClassification"
        )

        super().__init__(config)

        if config.is_decoder:
            raise ValueError("This model cannot be used as decoder")

        self.deberta = DebertaV2Model(config)

        self.pooler = ContextPooler(config)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.classifier = ExtendedClassificationHead(config)

        self.post_init()

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
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the DeBERTa loss. Input should be a sequence of tokens
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        """

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

        sequence_output = discriminator_hidden_states.last_hidden_state

        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.flatten())

        return SeqClassOutput(
            last_hidden_state=sequence_output,
            hidden_states=sequence_output.hidden_states,
            attentions=sequence_output.attentions,
            seq_class_loss=loss,
            seq_class_logits=logits,  # (batch_size, k, num_labels)
        )


class DebertaV2ForPreTrainingAndExtendedSequenceClassification(DebertaV2PreTrainedModel):

    def __init__(self, config: DebertaV2ExtendedConfig):
        assert isinstance(config, DebertaV2ExtendedConfig), (
            "ExtendedConfig is required for ExtendedSequenceClassification"
        )

        super().__init__(config)

        if config.is_decoder:
            raise ValueError("This model cannot be used as decoder")

        self.deberta = DebertaV2Model(config)
        self.add_special_embedding_layer()
        self.discriminator_predictions = DebertaDiscriminatorPredictions(config)

        self.pooler = ContextPooler(config)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.classifier = ExtendedClassificationHead(config)

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
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the DeBERTa loss. Input should be a sequence of tokens
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        """

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

        classification_logits = self.classifier(pooled_output)

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

        classification_loss = None
        if seq_class_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
            classification_loss = loss_fct(
                classification_logits.view(-1, self.config.num_labels), seq_class_labels.flatten()
            )

        return TokenDetectionAndSeqClassOutput(
            last_hidden_state=discriminator_sequence_output,
            seq_class_loss=classification_loss,
            seq_class_logits=classification_logits,
            token_detection_loss=td_loss,
            token_detection_logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
