from typing import List, Union

from transformers import PretrainedConfig


class SummaryformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~Summaryformer`.
    It is used to instantiate a Summaryformer model according to the specified
    arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
    `roberta-base <https://huggingface.co/roberta-base>`__ architecture with a sequence length 4,096.

    The :class:`~SummaryformerConfig` class directly inherits :class:`~transformers.RobertaConfig`. It reuses
    the same defaults. Please check the parent class for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        local_attention_window (`int`, *optional*, defaults to 32):
            Size of the windows for the local attention.
            Takes into account both left and right tokens.
        local_attention_stride (`int`, *optional*, defaults to 0):
            Stride of the windows for the local attention.
        pad_token_id (`int`, *optional*, defaults to 1): Pad token id.
        bos_token_id (`int`, *optional*, defaults to 0): Begin of sentence token id.
        eos_token_id (`int`, *optional*, defaults to 2): End of sentence token id.

    Example::

        >>> from transformers import SummaryformerConfig, SummaryformerModel

        >>> # Initializing a Summaryformer configuration
        >>> configuration = SummaryformerConfig()

        >>> # Initializing a model from the configuration
        >>> model = SummaryformerModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "summaryformer"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: int = 0.1,
        attention_probs_dropout_prob: int = 0.1,
        max_position_embeddings: int = 4096,
        type_vocab_size: int = 2,
        initializer_range: int = 0.02,
        layer_norm_eps: int = 1e-12,
        use_cache: bool = True,
        classifier_dropout: bool = None,
        position_embedding_type: str = "absolute",
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        local_attention_window: int = 32,
        summary_attention_window: Union[int, List[int]] = 32,
        summary_attention_stride: Union[int, List[int]] = 0,
        **kwargs,
    ):
        r""" Constructs SummaryFormerConfig. """
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        assert local_attention_window > 0, "`local_attention_window` has to be greater than 0"
        assert local_attention_window % 2 == 0, "`local_attention_window` has to be an even number"

        assert isinstance(summary_attention_window, (int, list, tuple)), (
            "`summary_attention_window` must be integer or integer sequence"
        )
        assert isinstance(summary_attention_stride, (int, list, tuple)), (
            "`summary_attention_stride` must be integer or integer sequence"
        )

        if isinstance(summary_attention_window, int):
            summary_attention_window = [summary_attention_window] * num_hidden_layers
        if isinstance(summary_attention_stride, int):
            summary_attention_stride = [summary_attention_stride] * num_hidden_layers

        assert all(s >= 0 for s in summary_attention_window)
        assert all(s >= 0 for s in summary_attention_stride)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.position_embedding_type = position_embedding_type
        self.local_attention_window = local_attention_window
        self.summary_attention_window = summary_attention_window
        self.summary_attention_stride = summary_attention_stride
