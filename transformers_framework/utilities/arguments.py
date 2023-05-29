from argparse import Action, ArgumentError, ArgumentParser, Namespace
from typing import Any, Dict, List, Union

from transformers_framework.utilities.initilization import initialize_precision, initialize_strategy
from transformers_framework.utilities.logging import rank_zero_warn


class FlexibleArgumentParser(ArgumentParser):

    def add_argument(self, *args, exist_ok: bool = True, replace: bool = False, **kwargs) -> Action:
        if exist_ok:
            try:
                return super().add_argument(*args, **kwargs)
            except ArgumentError as e:
                rank_zero_warn(f"Argument {e.argument_name} was define twice, make sure this is intended...")
                if replace:
                    self.remove_options(args)
                    return super().add_argument(*args, **kwargs)
        else:
            return super().add_argument(*args, **kwargs)

    def remove_options(self, options: List[str]):
        for option in options:
            for action in self._actions:
                if vars(action)['option_strings'][0] == option:
                    self._handle_conflict_resolve(None, [(option, action)])
                    break

    def add_argument_if(self, sentinel, value, *args, **kwargs):
        parameters, _ = self.parse_known_args()
        parameters = vars(parameters)
        if sentinel not in parameters:
            rank_zero_warn(f"Argument {sentinel} was used as sentinel, but was not defined in parser yet...")
        elif parameters[sentinel] == value:
            self.add_argument(*args, **kwargs)

    def add_argument_if_not(self, sentinel, value, *args, **kwargs):
        parameters, _ = self.parse_known_args()
        parameters = vars(parameters)
        if sentinel not in parameters:
            rank_zero_warn(f"Argument {sentinel} was used as sentinel, but was not defined in parser yet...")
        elif parameters[sentinel] != value:
            self.add_argument(*args, **kwargs)


def int_non_negative(value: Any) -> int:
    r""" Check and return non-negative integer. """
    value = int(value)
    if value < 0:
        raise ValueError
    return value


def int_positive(value: Any) -> int:
    r""" Check and return positive integer. """
    value = int(value)
    if value <= 0:
        raise ValueError
    return value


def float_non_negative(value: Any) -> float:
    r""" Check and return non-negative float. """
    value = float(value)
    if value < 0.0:
        raise ValueError
    return value


def float_positive(value: Any) -> float:
    r""" Check and return positive float. """
    value = float(value)
    if value <= 0.0:
        raise ValueError
    return value


def float_greater(lower: float):
    r""" Check value is float and `lower` < value. """
    def return_fn(value: Any):
        value = float(value)
        if value <= lower:
            raise ValueError
        return value
    return return_fn


def float_greater_or_equal(lower: float):
    r""" Check value is float and `lower` <= value. """
    def return_fn(value: Any):
        value = float(value)
        if value < lower:
            raise ValueError
        return value
    return return_fn


def float_less(upper: float):
    r""" Check value is float and value < `upper`. """
    def return_fn(value: Any):
        value = float(value)
        if value >= upper:
            raise ValueError
        return value
    return return_fn


def float_less_or_equal(upper: float):
    r""" Check value is float and value <= `upper`. """
    def return_fn(value: Any):
        value = float(value)
        if value > upper:
            raise ValueError
        return value
    return return_fn


def int_greater(lower: int):
    r""" Check value is int and `lower` < value. """
    def return_fn(value: Any):
        value = int(value)
        if value <= lower:
            raise ValueError
        return value
    return return_fn


def int_less(upper: int):
    r""" Check value is int and value < `upper`. """
    def return_fn(value: Any):
        value = int(value)
        if value >= upper:
            raise ValueError
        return value
    return return_fn


def float_between(lower: float, upper: float):
    r""" Check value is float and `lower` <= value <= `upper` (both ended inclusive). """
    def return_fn(value: Any):
        value = float(value)
        if not (lower <= value <= upper):
            raise ValueError
        return value
    return return_fn


def int_between(lower: int, upper: int):
    r""" Check value is int and `lower` <= value < `upper` (open interval on right). """
    def return_fn(value: Any):
        value = int(value)
        if not (lower <= value < upper):
            raise ValueError
        return value
    return return_fn


def int_or_float(value: Any) -> Union[int, float]:
    try:
        return int(value)
    except ValueError:
        return float(value)


def int_or_str(value: Any) -> Union[int, str]:
    try:
        return int(value)
    except ValueError:
        return str(value)


def is_already_defined_in_argparse(parser: ArgumentParser, name: str) -> bool:
    r""" Check if argument `name` has already been defined in parser. """
    for action in parser._actions:
        if name == action.dest:
            return True
    return False


def add_seq_class_arguments(parser: ArgumentParser):
    r""" Add default arguments for simple sequence classification. """
    parser.add_argument('--num_labels', type=int, default=2, required=False)


def add_token_class_arguments(parser: ArgumentParser):
    r""" Add default arguments for simple token classification. """
    parser.add_argument('--num_labels', type=int, default=2, required=False)


def add_answer_selection_arguments(parser: ArgumentParser):
    r""" Add default AS2 arguments. """
    parser.add_argument(
        '--metrics_empty_target_action',
        choices=('skip', 'neg', 'pos', 'error'),
        default='skip',
        required=False,
        help="Empty target action for test metrics",
    )


def add_question_answering_arguments(parser: ArgumentParser):
    r""" Add default MR arguments. """
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=(
            "The maximum number of tokens for the question. "
            "Questions longer than this will be truncated to this length."
        )
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start and "
            "end predictions are not conditioned on one another."
        )
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate for machine reading."
    )


def add_summarization_arguments(parser: ArgumentParser):
    r""" Add default summarization arguments. """
    add_generation_arguments(parser)


def add_multi_token_summarization_arguments(parser: ArgumentParser):
    add_summarization_arguments(parser)
    parser.add_argument(
        '--max_multi_token_predictions', type=int, required=True, help="Max number of tokens in span"
    )
    parser.add_argument(
        '--whole_word_summarization', action="store_true", help="Force masking only of complete words"
    )


def add_token_detection_arguments(parser: ArgumentParser):
    r""" Add default token detection arguments. """
    parser.add_argument(
        '--probability',
        type=float_between(lower=0.0, upper=1.0),
        default=0.15,
        help="Probability of replacing a token",
    )


def add_random_token_detection_arguments(parser: ArgumentParser):
    r""" Add default random token detection arguments. """
    add_token_detection_arguments(parser)
    parser.add_argument('--whole_word_detection', action="store_true", help="Enables whole word replacing")


def add_masked_lm_arguments(parser: ArgumentParser):
    r""" Add MLM arguments. """
    parser.add_argument(
        '--probability', type=float_between(lower=0.0, upper=1.0), default=0.15, help="Probability of changing a token"
    )
    parser.add_argument(
        '--probability_masked',
        type=float_between(lower=0.0, upper=1.0),
        default=0.80,
        help="Sub-probability of masking a token when selected",
    )
    parser.add_argument(
        '--probability_replaced',
        type=float_between(lower=0.0, upper=1.0),
        default=0.10,
        help="Sub-probability of replacing a token when selected"
    )
    parser.add_argument(
        '--probability_unchanged',
        type=float_between(lower=0.0, upper=1.0),
        default=0.10,
        help="Sub-probability of leaving unchanged a token when selected"
    )
    parser.add_argument('--whole_word_masking', action="store_true", help="Enables whole word masking")
    # check sum of probabilities is 1.0
    tmp_args, _ = parser.parse_known_args()
    if sum([tmp_args.probability_masked, tmp_args.probability_replaced, tmp_args.probability_unchanged]) != 1:
        raise ValueError(
            "The sum of `probability_masked`, `probability_replaced` and `probability_unchanged` must be 1.0"
        )


def add_denoising_arguments(parser: ArgumentParser):
    r""" Add denoising parameters. """
    parser.add_argument(
        '--probability',
        type=float_between(lower=0.0, upper=1.0),
        required=True,
        help="probability of denoising a token",
    )
    parser.add_argument('--mean_span_length', type=float, required=True, help="Average span length")
    parser.add_argument(
        '--max_number_of_spans', type=int_positive, default=None, help="Maximum number of span masked"
    )
    parser.add_argument(
        '--whole_word_denoising', action="store_true", help="Force masking only of complete words"
    )


def add_multi_token_arguments(parser: ArgumentParser):
    r""" Arguments for multi-token generative models. """
    parser.add_argument(
        '--max_multi_token_predictions', type=int_greater(0), required=True, help="Max number of tokens in span"
    )


def add_masked_lm_and_token_detection_arguments(parser: ArgumentParser):
    r""" Add MLM and TD arguments. """
    parser.add_argument('--sample_function', type=str, default='gumbel', choices=['gumbel', 'multinomial'])
    parser.add_argument('--pre_trained_generator_config', type=str, default=None)
    parser.add_argument('--pre_trained_generator_model', type=str, default=None)
    parser.add_argument('--tie_generator_discriminator_embeddings', action="store_true")
    parser.add_argument('--generator_size', type=float, default=1 / 2)


def add_generation_arguments(parser: ArgumentParser):
    r""" Add generation arguments. """
    parser.add_argument('--generation_min_length', type=int, default=None, required=False)
    parser.add_argument('--generation_max_length', type=int, default=None, required=False)
    parser.add_argument('--generation_min_new_tokens', type=int, default=None, required=False)
    parser.add_argument('--generation_max_new_tokens', type=int, default=None, required=False)
    parser.add_argument('--generation_do_sample', action="store_true")
    parser.add_argument('--generation_early_stopping', action="store_true")
    parser.add_argument('--generation_num_beams', type=int, default=1, required=False)
    parser.add_argument('--generation_temperature', type=float, default=None, required=False)
    parser.add_argument('--generation_top_k', type=int, default=None, required=False)
    parser.add_argument('--generation_top_p', type=float, default=None, required=False)
    parser.add_argument('--generation_typical_p', type=float, default=None, required=False)
    parser.add_argument('--generation_repetition_penalty', type=float, default=None, required=False)
    parser.add_argument('--generation_length_penalty', type=float, default=None, required=False)
    parser.add_argument('--generation_num_return_sequences', type=int, default=1, required=False)
    parser.add_argument('--generation_num_beam_groups', type=int, default=1, required=False)
    parser.add_argument('--generation_diversity_penalty', type=float, default=None, required=False)
    parser.add_argument('--generation_penalty_alpha', type=float, default=None, required=False)
    parser.add_argument('--generation_bad_words_ids', type=int, nargs='+', default=None, required=False)
    parser.add_argument('--generation_force_words_ids', type=int, nargs='+', default=None, required=False)
    parser.add_argument('--generation_no_repeat_ngram_size', type=int, default=None, required=False)
    parser.add_argument('--generation_encoder_no_repeat_ngram_size', type=int, default=None, required=False)
    parser.add_argument('--generation_max_time', type=float, default=None, required=False)
    parser.add_argument('--generation_renormalize_logits', action="store_true")
    parser.add_argument('--generation_suppress_tokens', type=int, nargs='+', default=None, required=False)
    parser.add_argument('--generation_begin_suppress_tokens', type=int, nargs='+', default=None, required=False)
    parser.add_argument('--generation_forced_decoder_ids', type=int, nargs='+', default=None, required=False)


def get_generation_args_from_hyperparameters(hyperparameters: Namespace) -> Dict:
    r""" Just extract generation hyperparameters from namespace. """
    res = dict(
        min_length=hyperparameters.generation_min_length,
        max_length=hyperparameters.generation_max_length,
        min_new_tokens=hyperparameters.generation_min_new_tokens,
        max_new_tokens=hyperparameters.generation_max_new_tokens,
        do_sample=hyperparameters.generation_do_sample,
        early_stopping=hyperparameters.generation_early_stopping,
        num_beams=hyperparameters.generation_num_beams,
        temperature=hyperparameters.generation_temperature,
        penalty_alpha=hyperparameters.generation_penalty_alpha,
        top_k=hyperparameters.generation_top_k,
        top_p=hyperparameters.generation_top_p,
        typical_p=hyperparameters.generation_typical_p,
        repetition_penalty=hyperparameters.generation_repetition_penalty,
        bad_words_ids=hyperparameters.generation_bad_words_ids,
        force_words_ids=hyperparameters.generation_force_words_ids,
        length_penalty=hyperparameters.generation_length_penalty,
        no_repeat_ngram_size=hyperparameters.generation_no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=hyperparameters.generation_encoder_no_repeat_ngram_size,
        num_return_sequences=hyperparameters.generation_num_return_sequences,
        max_time=hyperparameters.generation_max_time,
        num_beam_groups=hyperparameters.generation_num_beam_groups,
        diversity_penalty=hyperparameters.generation_diversity_penalty,
        renormalize_logits=hyperparameters.generation_renormalize_logits,
        suppress_tokens=hyperparameters.generation_suppress_tokens,
        begin_suppress_tokens=hyperparameters.generation_begin_suppress_tokens,
        forced_decoder_ids=hyperparameters.generation_forced_decoder_ids,
        output_attentions=False,
        output_hidden_states=False,
        output_scores=False,
        remove_invalid_values=True,
        return_dict_in_generate=True,
    )
    # remove None keys such that they are not passed to the generate method
    for k in list(res.keys()):
        if res[k] is None:
            del res[k]

    return res


def add_trainer_args(parser: ArgumentParser):

    parser.add_argument('--accelerator', type=str, default="auto", required=False)
    parser.add_argument('--strategy', type=str, default="auto", required=False)
    parser.add_argument('--devices', type=int, default="auto", required=False)
    parser.add_argument('--num_nodes', type=int, default=1, required=False)
    parser.add_argument('--precision', type=int_or_str, default=16, required=False, choices=(16, 'bf16', 32, 64))
    parser.add_argument('--fast_dev_run', action="store_true")
    parser.add_argument('--max_epochs', type=int, default=None, required=False)
    parser.add_argument('--min_epochs', type=int, default=None, required=False)
    parser.add_argument('--max_steps', type=int, default=-1, required=False)
    parser.add_argument('--min_steps', type=int, default=None, required=False)
    parser.add_argument('--limit_train_batches', type=int_or_float, default=None, required=False)
    parser.add_argument('--limit_val_batches', type=int_or_float, default=None, required=False)
    parser.add_argument('--limit_test_batches', type=int_or_float, default=None, required=False)
    parser.add_argument('--limit_predict_batches', type=int_or_float, default=None, required=False)
    parser.add_argument('--val_check_interval', type=int_or_float, default=None, required=False)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, required=False)
    parser.add_argument('--num_sanity_val_steps', type=int, default=2, required=False)
    parser.add_argument('--log_every_n_steps', type=int, default=50, required=False)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, required=False)
    parser.add_argument('--gradient_clip_val', type=float, default=None, required=False)
    parser.add_argument('--gradient_clip_algorithm', type=str, default=None, required=False)
    parser.add_argument('--deterministic', action="store_true")
    parser.add_argument('--benchmark', action="store_true")
    parser.add_argument('--profiler', type=str, default=None, required=False)
    parser.add_argument('--reload_dataloaders_every_n_epochs', type=int, default=0, required=False)


def get_trainer_args_from_hyperparameters(hyperparameters: Namespace) -> Dict:
    r""" Just extract generation hyperparameters from namespace. """

    # strategies and precision setup
    strategy = initialize_strategy(hyperparameters)
    precision = initialize_precision(hyperparameters)

    res = dict(
        accelerator=hyperparameters.accelerator,
        strategy=strategy,
        devices=hyperparameters.devices,
        num_nodes=hyperparameters.num_nodes,
        precision=precision,
        fast_dev_run=hyperparameters.fast_dev_run,
        max_epochs=hyperparameters.max_epochs,
        min_epochs=hyperparameters.min_epochs,
        max_steps=hyperparameters.max_steps,
        min_steps=hyperparameters.min_steps,
        limit_train_batches=hyperparameters.limit_train_batches,
        limit_val_batches=hyperparameters.limit_val_batches,
        limit_test_batches=hyperparameters.limit_test_batches,
        limit_predict_batches=hyperparameters.limit_predict_batches,
        val_check_interval=hyperparameters.val_check_interval,
        check_val_every_n_epoch=hyperparameters.check_val_every_n_epoch,
        num_sanity_val_steps=hyperparameters.num_sanity_val_steps,
        log_every_n_steps=hyperparameters.log_every_n_steps,
        accumulate_grad_batches=hyperparameters.accumulate_grad_batches,
        gradient_clip_val=hyperparameters.gradient_clip_val,
        gradient_clip_algorithm=hyperparameters.gradient_clip_algorithm,
        deterministic=hyperparameters.deterministic,
        benchmark=hyperparameters.benchmark,
        profiler=hyperparameters.profiler,
        reload_dataloaders_every_n_epochs=hyperparameters.reload_dataloaders_every_n_epochs,
        enable_progress_bar=True,
        enable_model_summary=True,
        inference_mode=True,
        use_distributed_sampler=True,
    )

    return res
