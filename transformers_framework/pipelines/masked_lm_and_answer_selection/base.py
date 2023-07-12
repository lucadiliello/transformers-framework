from typing import Any, Dict, Union

from torchmetrics.classification.accuracy import BinaryAccuracy, MulticlassAccuracy
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
)
from transformers.configuration_utils import PretrainedConfig

from transformers_framework.architectures.modeling_outputs import MaskedLMAndSeqClassOutput
from transformers_framework.interfaces.logging import (
    ANSWER_SELECTION_HR_5,
    ANSWER_SELECTION_MAP,
    ANSWER_SELECTION_MRR,
    ANSWER_SELECTION_NDCG,
    ANSWER_SELECTION_P_1,
    LOSS,
    MASKED_LM_ACCURACY,
    MASKED_LM_LOSS,
    MASKED_LM_PERPLEXITY,
    SEQ_CLASS_ACCURACY,
    SEQ_CLASS_LOSS,
)
from transformers_framework.interfaces.step import MaskedLMAndAnswerSelectionStepOutput
from transformers_framework.metrics.perplexity import Perplexity
from transformers_framework.pipelines.pipeline.pipeline import ExtendedPipeline
from transformers_framework.processing.postprocessors import masked_lm_and_answer_selection_processor
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import (
    FlexibleArgumentParser,
    add_answer_selection_arguments,
    add_masked_lm_arguments,
)
from transformers_framework.utilities.torch import combine_losses


class MaskedLMAndAnswerSelectionPipeline(ExtendedPipeline):

    POST_FORWARD_ADAPTER = None  # models designed from MLM + AS2 return already a compatible object
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels')
    ]

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        # needed till full support for multiple classification is implemented
        assert self.hyperparameters.k is None

        # train metrics
        self.train_acc = BinaryAccuracy()
        self.train_mlm_acc = MulticlassAccuracy(self.tokenizer.vocab_size, average='micro', ignore_index=IGNORE_IDX)
        self.train_mlm_ppl = Perplexity(ignore_index=IGNORE_IDX)

        # validation metrics
        metrics_kwargs = dict(
            empty_target_action=hyperparameters.metrics_empty_target_action,
            compute_on_step=False,
            ignore_idx=IGNORE_IDX,
        )
        self.valid_acc = BinaryAccuracy()
        self.valid_map = RetrievalMAP(**metrics_kwargs)
        self.valid_mrr = RetrievalMRR(**metrics_kwargs)
        self.valid_p1 = RetrievalPrecision(k=1, **metrics_kwargs)
        self.valid_hr5 = RetrievalHitRate(k=5, **metrics_kwargs)
        self.valid_ndgc = RetrievalNormalizedDCG(**metrics_kwargs)

        # test metrics
        self.test_acc = BinaryAccuracy()
        self.test_map = RetrievalMAP(**metrics_kwargs)
        self.test_mrr = RetrievalMRR(**metrics_kwargs)
        self.test_p1 = RetrievalPrecision(k=1, **metrics_kwargs)
        self.test_hr5 = RetrievalHitRate(k=5, **metrics_kwargs)
        self.test_ndgc = RetrievalNormalizedDCG(**metrics_kwargs)

    def requires_extended_tokenizer(self):
        return len(self.hyperparameters.input_columns) > 2 or self.hyperparameters.extended_token_type_ids is not None

    def requires_extended_model(self):
        return self.hyperparameters.k is not None

    def configure_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        kwargs['num_labels'] = 2  # always 2 classes for answer selection
        return super().configure_config(**kwargs)

    def step(self, batch: Dict) -> MaskedLMAndAnswerSelectionStepOutput:
        r""" Forward step is shared between all train/val/test steps. """

        index = batch.pop('index')
        results: MaskedLMAndSeqClassOutput = self.forward(**batch)

        # reranking
        seq_class_scores = results.seq_class_logits.softmax(dim=-1)[:, -1]
        seq_class_predictions = results.seq_class_logits.argmax(dim=-1)

        # Loss is linear combination of MLM and classification losses
        loss = combine_losses(
            losses=[results.masked_lm_loss, results.seq_class_loss],
            weigths=[self.hyperparameters.masked_lm_weight, self.hyperparameters.seq_class_weight],
        )

        return MaskedLMAndAnswerSelectionStepOutput(
            loss=loss,
            masked_lm_loss=results.masked_lm_loss,
            masked_lm_predictions=results.masked_lm_logits.argmax(dim=-1),
            masked_lm_logits=results.masked_lm_logits,
            masked_lm_labels=batch['masked_lm_labels'],
            seq_class_index=index,
            seq_class_loss=results.seq_class_loss,
            seq_class_predictions=seq_class_predictions,
            seq_class_scores=seq_class_scores,
            seq_class_labels=batch['seq_class_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        train_mlm_acc = self.train_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        train_mlm_ppl = self.train_mlm_ppl(step_output.masked_lm_logits.float(), step_output.masked_lm_labels)
        train_acc = self.train_acc(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, train_acc)
        self.log(MASKED_LM_LOSS, step_output.masked_lm_loss)
        self.log(MASKED_LM_ACCURACY, train_mlm_acc)
        self.log(MASKED_LM_PERPLEXITY, train_mlm_ppl)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output = self.step(batch)

        # AS2 metrics should only be computed globally
        self.validation_step_update_metrics(step_output)

        valid_acc = self.valid_acc(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, valid_acc)

    def validation_step_update_metrics(self, step_output: MaskedLMAndAnswerSelectionStepOutput):
        r""" Update metrics for answer selection. """
        kwargs = dict(
            preds=step_output.seq_class_scores,
            target=step_output.seq_class_labels,
            indexes=step_output.seq_class_index,
        )
        self.valid_map.update(**kwargs)
        self.valid_mrr.update(**kwargs)
        self.valid_p1.update(**kwargs)
        self.valid_hr5.update(**kwargs)
        self.valid_ndgc.update(**kwargs)

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output = self.step(batch)

        # AS2 metrics should only be computed globally
        self.test_step_update_metrics(step_output)

        test_acc = self.test_acc(step_output.seq_class_predictions, step_output.seq_class_labels)

        self.log(LOSS, step_output.loss)
        self.log(SEQ_CLASS_LOSS, step_output.seq_class_loss)
        self.log(SEQ_CLASS_ACCURACY, test_acc)

    def test_step_update_metrics(self, step_output: MaskedLMAndAnswerSelectionStepOutput):
        r""" Update metrics for answer selection. """
        kwargs = dict(
            preds=step_output.seq_class_scores,
            target=step_output.seq_class_labels,
            indexes=step_output.seq_class_index,
        )
        self.test_map.update(**kwargs)
        self.test_mrr.update(**kwargs)
        self.test_p1.update(**kwargs)
        self.test_hr5.update(**kwargs)
        self.test_ndgc.update(**kwargs)

    def on_validation_epoch_end(self):
        r""" Just log metrics. """
        super().on_validation_epoch_end()
        self.log(ANSWER_SELECTION_MAP, self.valid_map.compute())
        self.log(ANSWER_SELECTION_MRR, self.valid_mrr.compute())
        self.log(ANSWER_SELECTION_P_1, self.valid_p1.compute())
        self.log(ANSWER_SELECTION_HR_5, self.valid_hr5.compute())
        self.log(ANSWER_SELECTION_NDCG, self.valid_ndgc.compute())

    def on_test_epoch_end(self):
        r""" Just log metrics. """
        super().on_test_epoch_end()
        self.log(ANSWER_SELECTION_MAP, self.test_map.compute())
        self.log(ANSWER_SELECTION_MRR, self.test_mrr.compute())
        self.log(ANSWER_SELECTION_P_1, self.test_p1.compute())
        self.log(ANSWER_SELECTION_HR_5, self.test_hr5.compute())
        self.log(ANSWER_SELECTION_NDCG, self.test_ndgc.compute())

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add denoising objective. """
        return masked_lm_and_answer_selection_processor(
            sample=sample,
            input_columns=self.hyperparameters.input_columns,
            index_column=self.hyperparameters.index_column,
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
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--input_columns', type=str, nargs='+', required=True)
        parser.add_argument('--label_column', type=str, required=True)
        parser.add_argument('--index_column', type=str, required=True)
        parser.add_argument('--seq_class_weight', type=float, default=1.0)
        parser.add_argument('--masked_lm_weight', type=float, default=1.0)
        add_masked_lm_arguments(parser)
        add_answer_selection_arguments(parser)
