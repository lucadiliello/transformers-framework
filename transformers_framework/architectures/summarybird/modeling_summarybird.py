import math
from typing import Optional

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from transformers.activations import ACT2FN, gelu
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.longt5.modeling_longt5 import (
    _concatenate_3_blocks,
    _get_local_attention_mask,
    _split_into_blocks,
)

from transformers_framework.architectures.modeling_outputs import MaskedLMAndTokenClassOutput
from transformers_framework.architectures.summarybird.configuration_summarybird import SummaryBirdConfig
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.models import (
    compute_relative_position_bucket,
    create_position_ids_from_input_ids,
    create_position_ids_from_inputs_embeds,
    get_summary_attention_mask,
    make_global_fixed_block_ids,
    make_side_relative_position_ids,
)


class SummaryBirdSelfOutput(nn.Module):

    def __init__(self, config: SummaryBirdConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SummaryBirdIntermediate(nn.Module):

    def __init__(self, config: SummaryBirdConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = (
            ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SummaryBirdOutput(nn.Module):

    def __init__(self, config: SummaryBirdConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SummaryBirdEmbeddings(nn.Module):
    r""" Same as BertEmbeddings with a tiny tweak for positional embeddings indexing. """

    def __init__(self, config: SummaryBirdConfig):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, 0)
            else:
                position_ids = create_position_ids_from_inputs_embeds(inputs_embeds, self.padding_idx)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros,
        # which usually occurs when its auto-generated, registered buffer helps users when tracing the
        # model without passing token_type_ids, solves issue #5664
        if token_type_ids is None:
            buffered_token_type_ids = self.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SummaryBirdTransientGlobalAttention(nn.Module):

    def __init__(self, config: SummaryBirdConfig, has_relative_attention_bias: bool = False):
        super().__init__()

        self.config = config

        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_attention_heads)

        self.pruned_heads = set()

        self.local_block_size = config.local_block_size  # for local attention
        self.summaries_block_size = config.summaries_block_size  # for global summaries

        if self.has_relative_attention_bias:
            self.global_relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.num_attention_heads
            )

        self.global_input_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def shape(self, states):
        r""" Projection (batch_size, seq_length, hidden_size) ->
        (batch_size, seq_length, num_attention_heads, attention_head_size).
        """
        return states.view(states.shape[0], -1, self.num_attention_heads, self.attention_head_size)

    def unshape(self, states):
        r""" Projection (batch_size, seq_length, num_attention_heads, attention_head_size) ->
        (batch_size, seq_length, hidden_size).
        """
        return states.contiguous().view(states.shape[0], -1, self.all_head_size)

    def compute_full_bias(self, length: int) -> torch.Tensor:
        r""" Compute binned relative position bias.

        Relative position example with length=6:
            # relative_position:
                tensor([
                    [ 0,  1,  2,  3,  4,  5],
                    [-1,  0,  1,  2,  3,  4],
                    [-2, -1,  0,  1,  2,  3],
                    [-3, -2, -1,  0,  1,  2],
                    [-4, -3, -2, -1,  0,  1],
                    [-5, -4, -3, -2, -1,  0],
                ])
            Then positions are translated to corresponding buckets and encoded to head size.

        Returns
            a tensor of shape (num_attention_heads, length, length)
        """
        device = self.relative_attention_bias.weight.device

        context_position = memory_position = torch.arange(length, dtype=torch.long, device=device)
        relative_position = memory_position[None, :] - context_position[:, None]
        # matrix with integers representing the distance between every pair of tokens

        relative_position_bucket = compute_relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # embed positions
        values = self.relative_attention_bias(relative_position_bucket)  # shape (length, length, num_attention_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)

        # shape (1, num_attention_heads, length, length)
        return values

    def compute_local_bias(self, length: int) -> torch.Tensor:
        r""" Compute binned relative position bias for 3 contiguous blocks.

        Relative position example with length=3 and full:
            # relative_position:
                tensor([
                    [-3, -2, -1,  0,  1,  2,  3,  4,  5],
                    [-4, -3, -2, -1,  0,  1,  2,  3,  4],
                    [-5, -4, -3, -2, -1,  0,  1,  2,  3],
                ])
            Then positions are translated to corresponding bucket and encoded to head size.

        Returns
            a tensor of shape (num_attention_heads, length, length)
        """
        device = self.relative_attention_bias.weight.device

        memory_position = torch.arange(3 * length, dtype=torch.long, device=device)
        context_position = memory_position[length:-length]

        relative_position = memory_position[None, :] - context_position[:, None]
        # shape (length, 3 * length)

        relative_position_bucket = compute_relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # shape (length, 3 * length, num_attention_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)

        # shape (1, 1, num_attention_heads, length, 3 * length)
        return values

    def compute_summary_bias(
        self, attention_mask: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        r""" Compute position bias only for summary attention.

        Returns
            a tensor of shape (batch_size, num_blocks, num_attention_heads, summaries_block_size, summaries_block_size)
        """

        # Compute local attention mask
        if attention_mask is not None:
            # We need to adjust position bias shape to be sum with mask
            local_attention_mask = get_summary_attention_mask(attention_mask, self.summaries_block_size, device)
            # Replace masked positions with -10_000 (according to the original implementation)
            local_attention_mask = torch.where(local_attention_mask > 0, 0.0, -1e4)
        else:
            local_attention_mask = None

        # position_bias shape: # (1, 1, num_attention_heads, summaries_block_size, summaries_block_size)
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, 1, self.num_attention_heads, self.summaries_block_size, self.summaries_block_size),
                device=device,
                dtype=dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_full_bias(self.summaries_block_size).unsqueeze(0)

        # local part
        if local_attention_mask is not None:
            # (batch_size, 1, num_attention_heads, summaries_block_size, summaries_block_size)
            position_bias = position_bias + local_attention_mask.transpose(1, 2)

        position_bias = position_bias.type(dtype)

        return position_bias

    def compute_side_bias(self, attention_mask: torch.Tensor, global_segment_ids: torch.Tensor) -> torch.Tensor:
        r""" Compute bias tensor for summarized part of the attention matrix. """
        # (batch_size, 1, seq_len, global_seq_len)
        side_attention_mask = torch.eq(attention_mask[..., None], global_segment_ids[:, None, :])[:, None, ...]

        attention_side_bias = torch.where(side_attention_mask > 0, 0.0, -1e4)
        # (batch_size, seq_len, global_seq_len)
        side_relative_position = make_side_relative_position_ids(attention_mask, self.summaries_block_size)

        side_relative_position_bucket = compute_relative_position_bucket(
            side_relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # (batch_size, seq_len, global_seq_len, num_attention_heads)
        side_bias = self.global_relative_attention_bias(side_relative_position_bucket)

        # (batch_size, num_attention_heads, seq_len, global_seq_len)
        side_bias = side_bias.permute([0, 3, 1, 2])
        attention_side_bias = attention_side_bias + side_bias

        # (batch_size, num_attention_heads, seq_len, global_seq_len)
        return attention_side_bias

    def compute_position_bias(
        self, attention_mask: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        r""" Compute position bias for local + summary global attention.
        
        Returns
            a tensor of shape (batch_size, num_blocks, num_attention_heads,
                local_block_size, 3 * local_block_size + global_seq_len)
        """

        batch_size, seq_length = attention_mask.shape[:2]

        # Compute local attention mask
        if attention_mask is not None:
            # We need to adjust position bias shape to be sum with mask
            local_attention_mask = _get_local_attention_mask(attention_mask, self.local_block_size, device)
            # Replace masked positions with -10_000 (according to the original implementation)
            local_attention_mask = torch.where(local_attention_mask > 0, 0.0, -1e4)
        else:
            local_attention_mask = None

        # position_bias shape: # (1, 1, n_heads, local_block_size, 3 * local_block_size)
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, 1, self.num_attention_heads, self.local_block_size, 3 * self.local_block_size),
                device=device,
                dtype=dtype,
            )
        else:
            position_bias = self.compute_local_bias(self.local_block_size)

        # local part
        if local_attention_mask is not None:
            # (batch_size, 1, n_heads, local_block_size, 3 * local_block_size)
            position_bias = position_bias + local_attention_mask.transpose(1, 2)

        position_bias = position_bias.type(dtype)

        # Calculate global/side bias - shape: # (batch_size, num_attention_heads, seq_len, global_seq_len)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length)

        _, global_segment_ids = make_global_fixed_block_ids(attention_mask, block_size=self.summaries_block_size)
        # global_segment_ids shape: (batch_size, global_seq_len)

        side_position_bias = self.compute_side_bias(attention_mask, global_segment_ids)

        # (batch_size, num_blocks, num_attention_heads, local_block_size, global_seq_len)
        side_position_bias = _split_into_blocks(side_position_bias, self.local_block_size, dim=-2).transpose(1, 2)
        side_position_bias = side_position_bias.to(device=device, dtype=dtype)

        # (batch_size, num_blocks, num_attention_heads, local_block_size, 3 * local_block_size + summaries_block_size)
        position_bias = torch.cat([position_bias, side_position_bias], dim=-1)
        return position_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_bias: Optional[torch.FloatTensor] = None,
        summary_position_bias: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        r""" Self-attention layer.
        hidden_states is (batch_size, seq_length, hidden_size)
        attention_mask is (batch_size, seq_length)
        """

        # ###############################
        # ### General projection part ###
        # ###############################

        # get normal/local states
        # shape: (batch_size, seq_length, num_attention_heads, dim_per_head)
        query_states = self.shape(self.query(hidden_states))
        key_states = self.shape(self.key(hidden_states))
        value_states = self.shape(self.value(hidden_states))

        # Split into blocks -> (batch_size, num_blocks, local_block_size, num_attention_heads, dim_per_head)
        local_query_states = _split_into_blocks(query_states, self.local_block_size, dim=1)
        local_key_states = _split_into_blocks(key_states, self.local_block_size, dim=1)
        local_value_states = _split_into_blocks(value_states, self.local_block_size, dim=1)

        # Split into blocks -> (batch_size, num_blocks, summaries_block_size, num_attention_heads, dim_per_head)
        summary_query_states = _split_into_blocks(query_states, self.summaries_block_size, dim=1)
        summary_key_states = _split_into_blocks(key_states, self.summaries_block_size, dim=1)
        summary_value_states = _split_into_blocks(value_states, self.summaries_block_size, dim=1)

        # ####################
        # ### Summary part ###
        # ####################

        # Compute scores -> (batch_size, num_blocks, num_attention_heads, summaries_block_size, summaries_block_size)
        summary_attention_scores = torch.einsum("...qhd,...khd->...hqk", summary_query_states, summary_key_states)
        summary_attention_scores = summary_attention_scores / math.sqrt(self.attention_head_size)

        # Compute global attention mask
        if summary_position_bias is None:
            summary_position_bias = self.compute_summary_bias(
                attention_mask, device=summary_attention_scores.device, dtype=summary_attention_scores.dtype
            )

        summary_attention_scores += summary_position_bias

        # Compute weights -> (batch_size, num_blocks, num_attention_heads, summaries_block_size, summaries_block_size)
        summary_attention_weights = nn.functional.softmax(
            summary_attention_scores.float(), dim=-1
        ).type_as(summary_attention_scores)
        summary_attention_weights = nn.functional.dropout(
            summary_attention_weights, p=self.config.attention_probs_dropout_prob, training=self.training
        )

        # Mask heads if we want to
        if head_mask is not None:
            summary_attention_weights = summary_attention_weights * head_mask

        # summary_attention_output
        # shape: (batch_size, num_blocks, summaries_block_size, num_attention_heads, dim_per_head)
        summary_attention_output = torch.einsum(
            "...hqk,...khd->...qhd", summary_attention_weights, summary_value_states
        )

        # summary_attention_output shape: (batch_size, num_blocks, hidden_size)
        # num_blocks == number of summary tokens
        summary_attention_output = summary_attention_output.mean(dim=2)
        summary_attention_output = summary_attention_output.view(
            *summary_attention_output.shape[:2], self.all_head_size
        )

        # global_inputs has shape (batch_size, num_blocks, hidden_size)
        summary_inputs = self.global_input_layer_norm(summary_attention_output)

        # ############################
        # ### Local + Summary part ###
        # ############################

        # Get global/side key/value states
        # shape: (batch_size, num_blocks, num_attention_heads, dim_per_head)
        side_key_states = self.shape(self.key(summary_inputs))
        side_value_states = self.shape(self.value(summary_inputs))

        # Concatenate 3 blocks for keys and values
        # -> (batch_size, num_blocks, 3 * local_block_size, num_attention_heads, dim_per_head)
        local_key_states = _concatenate_3_blocks(local_key_states, block_dim=1, sequence_dim=2)
        local_value_states = _concatenate_3_blocks(local_value_states, block_dim=1, sequence_dim=2)

        # Tile side inputs across local key/value blocks
        reps = [1] * (side_key_states.ndim + 1)
        reps[1] = local_key_states.shape[1]
        # New shape: (batch_size, num_blocks, num_blocks, num_attention_heads, dim_per_head)
        side_key_states = side_key_states.unsqueeze(1).repeat(reps)
        side_value_states = side_value_states.unsqueeze(1).repeat(reps)

        # Concatenate "local" and "side"/"global" key/value states to allow each token to attend global aggregated ones
        # New shape: (batch_size, num_blocks, 3 * local_block_size + num_blocks, num_attention_heads, dim_per_head)
        all_key_states = torch.cat([local_key_states, side_key_states], dim=2)
        all_value_states = torch.cat([local_value_states, side_value_states], dim=2)

        # Compute scores
        # (batch_size, num_block, num_attention_heads, local_block_size, 3 * local_block_size + num_blocks)
        attention_scores = torch.einsum("...qhd,...khd->...hqk", local_query_states, all_key_states)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Compute global attention mask
        if position_bias is None:
            position_bias = self.compute_position_bias(
                attention_mask, device=attention_scores.device, dtype=attention_scores.dtype
            )

        attention_scores += position_bias

        # (batch_size, num_blocks, num_attention_heads, local_block_size, 3 * local_block_size + num_blocks)
        attention_weights = nn.functional.softmax(attention_scores.float(), dim=-1).type_as(attention_scores)
        attention_weights = nn.functional.dropout(
            attention_weights, p=self.config.attention_probs_dropout_prob, training=self.training
        )

        # Mask heads if we want to
        if head_mask is not None:
            attention_weights = attention_weights * head_mask

        attention_output = self.unshape(torch.einsum("...hqk,...khd->...qhd", attention_weights, all_value_states))
        attention_output = attention_output[:, :hidden_states.shape[1], :]
        # remove eventual padding that was added for blocking

        outputs = (attention_output, position_bias, summary_position_bias)
        if output_attentions:
            outputs += (attention_weights, )

        return outputs  # (hidden_states, position_bias, summary_position_bias, attention_weights [Optional])


class SummaryBirdLayerTransientGlobalSelfAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.self = SummaryBirdTransientGlobalAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.output = SummaryBirdSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_bias: Optional[torch.FloatTensor] = None,
        summary_position_bias: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            summary_position_bias=summary_position_bias,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions and position biases if we output them
        return outputs


class SummaryBirdLayer(nn.Module):

    def __init__(self, config: SummaryBirdConfig, has_relative_attention_bias: bool = False):
        super().__init__()
        self.attention = SummaryBirdLayerTransientGlobalSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.intermediate = SummaryBirdIntermediate(config)
        self.output = SummaryBirdOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_bias: Optional[torch.FloatTensor] = None,
        summary_position_bias: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            summary_position_bias=summary_position_bias,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )  # (hidden_states, position_bias, attention_weights [Optional])

        hidden_states = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights and position biases

        # apply Feed Forward layer
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, hidden_states)

        outputs = (layer_output,) + outputs
        return outputs  # hidden_states, position bias, summary position bias, (self-attention weights)


class SummaryBirdEncoder(nn.Module):

    def __init__(self, config: SummaryBirdConfig):
        super().__init__()

        self.config = config
        self.layer = nn.ModuleList([
            SummaryBirdLayer(config, has_relative_attention_bias=bool(i == 0))
            for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None
        summary_position_bias = None

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    position_bias,
                    summary_position_bias,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    summary_position_bias=summary_position_bias,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )
            # layer_outputs is a tuple with: hidden_states, position_bias, (self-attention weights)

            # We share the position biases between the layers - the first layer store them
            hidden_states, position_bias, summary_position_bias = layer_outputs[:3]

            if output_attentions:
                attention_weights = layer_outputs[3]
                all_attentions = all_attentions + (attention_weights, )

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class SummaryBirdPreTrainedModel(PreTrainedModel):
    r""" An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models. """

    config_class = SummaryBirdConfig
    base_model_prefix = "summarybird"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        r""" Initialize the weights. """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value: bool = False):
        if isinstance(module, SummaryBirdEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        r""" Remove some keys from ignore list. """
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


class SummaryBirdPooler(nn.Module):

    def __init__(self, config: SummaryBirdConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SummaryBirdModel(SummaryBirdPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: SummaryBirdConfig, add_pooling_layer: bool = False):
        super().__init__(config)

        self.embeddings = SummaryBirdEmbeddings(config)
        self.encoder = SummaryBirdEncoder(config)

        self.pooler = SummaryBirdPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        r""" Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel. """

        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        r""" Returns: BaseModelOutput. """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length).to(device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_attention_heads] or [num_hidden_layers x num_attention_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_attention_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SummaryBirdLMHead(nn.Module):
    r""" As RoBERTa LM head."""

    def __init__(self, config: SummaryBirdConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class SummaryBirdForMaskedLM(SummaryBirdPreTrainedModel):

    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: SummaryBirdConfig):
        super().__init__(config)

        self.summarybird = SummaryBirdModel(config, add_pooling_layer=False)
        self.lm_head = SummaryBirdLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> MaskedLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100`
            are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        outputs = self.summarybird(
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
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SummaryBirdForMaskedLMAndTokenClassification(SummaryBirdPreTrainedModel):

    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: SummaryBirdConfig):
        super().__init__(config)

        self.summarybird = SummaryBirdModel(config, add_pooling_layer=False)
        self.lm_head = SummaryBirdLMHead(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        masked_lm_labels: Optional[torch.LongTensor] = None,
        token_class_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> MaskedLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100`
            are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        outputs = self.summarybird(
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
        prediction_scores = self.lm_head(sequence_output)
        token_class_logits = self.classifier(self.dropout(sequence_output))

        masked_lm_loss = None
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_IDX)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        token_class_loss = None
        if token_class_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_IDX)
            token_class_loss = loss_fct(
                token_class_logits.view(-1, self.config.num_labels), token_class_labels.view(-1)
            )

        return MaskedLMAndTokenClassOutput(
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            token_class_loss=token_class_loss,
            token_class_logits=token_class_logits,
            masked_lm_loss=masked_lm_loss,
            masked_lm_logits=prediction_scores,
        )


class SummaryBirdClassificationHead(nn.Module):
    r""" As RoBERTa sequence-level classification head."""

    def __init__(self, config: SummaryBirdConfig):
        super().__init__()
   
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states):
        res = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        res = self.dropout(res)
        res = self.dense(res)
        res = torch.tanh(res)
        res = self.dropout(res)
        res = self.out_proj(res)
        return res


class SummaryBirdForSequenceClassification(SummaryBirdPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.summarybird = SummaryBirdModel(config, add_pooling_layer=False)
        self.classifier = SummaryBirdClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
