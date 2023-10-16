from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaModel, RobertaPreTrainedModel

from transformers_framework.architectures.modeling_heads import ExtendedClassificationHead
from transformers_framework.architectures.modeling_outputs import MaskedLMAndSeqClassOutput, SeqClassOutput
from transformers_framework.architectures.roberta.configuration_roberta import RobertaExtendedConfig
from transformers_framework.utilities import IGNORE_IDX


class RobertaForExtendedSequenceClassification(RobertaPreTrainedModel):

    def __init__(self, config: RobertaExtendedConfig):
        assert isinstance(config, RobertaExtendedConfig), (
            "ExtendedConfig is required for ExtendedSequenceClassification"
        )

        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = ExtendedClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.flatten())

        return SeqClassOutput(
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            seq_class_loss=loss,
            seq_class_logits=logits,  # (batch_size, k, num_labels)
        )


class RobertaForMaskedLMAndExtendedSequenceClassification(RobertaPreTrainedModel):

    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: RobertaExtendedConfig):
        assert isinstance(config, RobertaExtendedConfig), (
            "ExtendedConfig is required for ExtendedSequenceClassification"
        )

        super().__init__(config)

        if config.is_decoder:
            raise ValueError("This model cannot be used as decoder")

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.classifier = ExtendedClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        masked_lm_labels=None,
        seq_class_labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        classification_logits = self.classifier(sequence_output)

        masked_lm_loss = None
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_IDX)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        classification_loss = None
        if seq_class_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
            classification_loss = loss_fct(
                classification_logits.view(-1, self.config.num_labels), seq_class_labels.flatten()
            )

        return MaskedLMAndSeqClassOutput(
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            seq_class_loss=classification_loss,
            seq_class_logits=classification_logits,
            masked_lm_loss=masked_lm_loss,
            masked_lm_logits=prediction_scores,
        )
