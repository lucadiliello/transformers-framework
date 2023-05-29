from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.deberta.modeling_deberta import (
    ContextPooler,
    DebertaModel,
    DebertaOnlyMLMHead,
    DebertaPreTrainedModel,
    StableDropout,
)

from transformers_framework.architectures.modeling_outputs import MaskedLMAndSeqClassOutput
from transformers_framework.utilities import IGNORE_IDX


class DebertaForMaskedLMAndSequenceClassification(DebertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaModel(config)

        # MLM
        self.cls = DebertaOnlyMLMHead(config)

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
