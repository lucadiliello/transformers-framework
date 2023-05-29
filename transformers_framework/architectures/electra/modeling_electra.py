from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers.models.electra.modeling_electra import (
    ElectraClassificationHead,
    ElectraConfig,
    ElectraDiscriminatorPredictions,
    ElectraModel,
    ElectraPreTrainedModel,
)

from transformers_framework.architectures.modeling_outputs import (
    TokenDetectionAndQuestionAnsweringOutput,
    TokenDetectionAndSeqClassOutput,
)
from transformers_framework.utilities import IGNORE_IDX


class ElectraForPreTrainingAndSequenceClassification(ElectraPreTrainedModel):

    def __init__(self, config: ElectraConfig):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.classifier = ElectraClassificationHead(config)

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
        token_detection_labels=None,
        seq_class_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)
        class_logits = self.classifier(discriminator_sequence_output)

        td_loss = None
        if token_detection_labels is not None:
            loss_fct = BCEWithLogitsLoss()
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
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
            seq_class_loss=class_loss,
            seq_class_logits=class_logits,
            token_detection_loss=td_loss,
            token_detection_logits=logits,
        )


class ElectraForPreTrainingAndQuestionAnswering(ElectraPreTrainedModel):

    def __init__(self, config: ElectraConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

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
        token_detection_labels=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)

        # Token Detection
        td_loss = None
        if token_detection_labels is not None:
            loss_fct = BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = token_detection_labels[active_loss]
                td_loss = loss_fct(active_logits, active_labels.float())
            else:
                td_loss = loss_fct(
                    logits.view(-1, discriminator_sequence_output.shape[1]), token_detection_labels.float()
                )

        # Question Answering
        qa_logits = self.qa_outputs(discriminator_sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        qa_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2

        return TokenDetectionAndQuestionAnsweringOutput(
            last_hidden_state=discriminator_sequence_output,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
            question_answering_loss=qa_loss,
            start_position_logits=start_logits,
            end_position_logits=end_logits,
            token_detection_loss=td_loss,
            token_detection_logits=logits,
        )
