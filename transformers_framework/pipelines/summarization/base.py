from typing import Any, Dict

from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score
from torchmetrics.text.sacre_bleu import SacreBLEUScore

from transformers_framework.architectures.modeling_outputs import SeqToSeqLMOutput
from transformers_framework.interfaces.adaptation import seq_to_seq_lm_adaptation
from transformers_framework.interfaces.logging import (
    LOSS,
    SEQ_TO_SEQ_LM_ACCURACY,
    SEQ_TO_SEQ_LM_F1,
    SEQ_TO_SEQ_LM_LOSS,
    SUMMARIZATION_BERTSCORE,
    SUMMARIZATION_BLEU,
    SUMMARIZATION_BLEURT,
    SUMMARIZATION_ROUGE,
)
from transformers_framework.interfaces.step import SeqToSeqGenStepOutput, SeqToSeqMaskedLMStepOutput
from transformers_framework.metrics.bert_score import BERTScore
from transformers_framework.metrics.bleurt import BLEURT
from transformers_framework.metrics.rouge import ROUGEScore
from transformers_framework.pipelines.pipeline import Pipeline
from transformers_framework.processing.postprocessors import summarization_processor
from transformers_framework.utilities.arguments import (
    FlexibleArgumentParser,
    add_summarization_arguments,
    get_generation_args_from_hyperparameters,
)


DEFAULT_BERTSCORE_MODEL = 'roberta-large'
DEFAULT_BLEURT_MODEL = 'lucadiliello/BLEURT-20'


class SummarizationPipeline(Pipeline):

    POST_FORWARD_ADAPTER = seq_to_seq_lm_adaptation
    MODEL_INPUT_NAMES_TO_REDUCE = [
        ('input_ids', 'attention_mask'),
        ('decoder_input_ids', 'decoder_attention_mask', 'seq_to_seq_lm_labels'),
    ]

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        if len(self.hyperparameters.max_sequence_length) != 2:
            raise ValueError(
                "Max sequence length should contain 2 values, one for the document and one for the summary"
            )

        metrics_args = (self.tokenizer.vocab_size, )
        metrics_kwargs = dict(average='micro', ignore_index=self.tokenizer.pad_token_id)

        # train metrics
        self.train_summarization_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.train_summarization_f1 = MulticlassF1Score(*metrics_args, **metrics_kwargs)

        # validation metrics
        self.valid_rouge = ROUGEScore(use_stemmer=True, name=SUMMARIZATION_ROUGE)
        self.valid_bleu = SacreBLEUScore(n_gram=4, smooth=True, tokenize='intl', lowercase=False)

        # test metrics
        self.test_rouge = ROUGEScore(use_stemmer=True, name=SUMMARIZATION_ROUGE)
        self.test_bleu = SacreBLEUScore(n_gram=4, smooth=True, tokenize='intl', lowercase=False)

        if hyperparameters.valid_bert_score:
            self.valid_bertscore = BERTScore(model_name_or_path=DEFAULT_BERTSCORE_MODEL, name=SUMMARIZATION_BERTSCORE)
        if hyperparameters.test_bert_score:
            self.test_bertscore = BERTScore(model_name_or_path=DEFAULT_BERTSCORE_MODEL, name=SUMMARIZATION_BERTSCORE)

        if hyperparameters.valid_bleurt_score:
            self.valid_bleurt = BLEURT(model_name_or_path=DEFAULT_BLEURT_MODEL)
        if hyperparameters.test_bleurt_score:
            self.test_bleurt = BLEURT(model_name_or_path=DEFAULT_BLEURT_MODEL)

        self.gen_kwargs = get_generation_args_from_hyperparameters(hyperparameters)

    def step(self, batch: Dict) -> SeqToSeqMaskedLMStepOutput:
        r""" Forward step is shared between all train/val/test steps. """
        batch.pop('original_summary')
        results: SeqToSeqLMOutput = self.forward(**batch)

        return SeqToSeqMaskedLMStepOutput(
            loss=results.seq_to_seq_lm_loss,
            seq_to_seq_lm_loss=results.seq_to_seq_lm_loss,
            seq_to_seq_lm_predictions=results.seq_to_seq_lm_logits.argmax(dim=-1),
            seq_to_seq_lm_labels=batch['seq_to_seq_lm_labels'],
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output = self.step(batch)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        train_summarization_acc = self.train_summarization_acc(
            step_output.seq_to_seq_lm_predictions, step_output.seq_to_seq_lm_labels
        )
        train_summarization_f1 = self.train_summarization_f1(
            step_output.seq_to_seq_lm_predictions, step_output.seq_to_seq_lm_labels
        )

        self.log(LOSS, step_output.loss)
        self.log(SEQ_TO_SEQ_LM_LOSS, step_output.seq_to_seq_lm_loss)
        self.log(SEQ_TO_SEQ_LM_ACCURACY, train_summarization_acc)
        self.log(SEQ_TO_SEQ_LM_F1, train_summarization_f1)

        return step_output.loss

    def evaluation_step(self, batch, *args) -> SeqToSeqGenStepOutput:
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        results = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            **self.gen_kwargs,
        )

        return SeqToSeqGenStepOutput(
            loss=None,
            generation_input_ids=results.sequences,
            generation_labels=batch['original_summary'],
        )

    def on_validation_epoch_start(self):
        r""" Load BERTScore model. """
        super().on_validation_epoch_start()
        if hasattr(self, 'valid_bertscore'):
            self.valid_bertscore.load()
        if hasattr(self, 'valid_bleurt'):
            self.valid_bleurt.load()

    def on_validation_epoch_end(self):
        r""" Unload BERTScore model. """
        super().on_validation_epoch_end()
        if hasattr(self, 'valid_bertscore'):
            self.valid_bertscore.unload()
        if hasattr(self, 'valid_bleurt'):
            self.valid_bleurt.unload()

    def validation_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the validation set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output = self.evaluation_step(batch)

        generated_sentences = self.tokenizer.batch_decode(
            step_output.generation_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        rouge_scores = self.valid_rouge(generated_sentences, step_output.generation_labels)  # rouge returns a dict
        bleu_scores = self.valid_bleu(generated_sentences, step_output.generation_labels)

        self.log(rouge_scores)
        self.log(SUMMARIZATION_BLEU, bleu_scores)

        if hasattr(self, 'valid_bertscore'):
            bert_score = self.valid_bertscore(generated_sentences, step_output.generation_labels)
            self.log(bert_score)
        if hasattr(self, 'valid_bleurt'):
            bleurt = self.valid_bleurt(generated_sentences, step_output.generation_labels)
            self.log(SUMMARIZATION_BLEURT, bleurt)

    def on_test_epoch_start(self):
        r""" Load BERTScore model. """
        super().on_test_epoch_start()
        if hasattr(self, 'test_bertscore'):
            self.test_bertscore.load()
        if hasattr(self, 'test_bleurt'):
            self.test_bleurt.load()

    def on_test_epoch_end(self):
        r""" Unload BERTScore model. """
        super().on_test_epoch_end()
        if hasattr(self, 'test_bertscore'):
            self.test_bertscore.unload()
        if hasattr(self, 'test_bleurt'):
            self.test_bleurt.unload()

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output = self.evaluation_step(batch)

        generated_sentences = self.tokenizer.batch_decode(
            step_output.generation_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        rouge_scores = self.test_rouge(generated_sentences, step_output.generation_labels)  # rouge returns a dict
        bleu_scores = self.test_bleu(generated_sentences, step_output.generation_labels)

        self.log(rouge_scores)
        self.log(SUMMARIZATION_BLEU, bleu_scores)

        if hasattr(self, 'test_bertscore'):
            bert_score = self.test_bertscore(generated_sentences, step_output.generation_labels)
            self.log(bert_score)
        if hasattr(self, 'test_bleurt'):
            bleurt = self.test_bleurt(generated_sentences, step_output.generation_labels)
            self.log(SUMMARIZATION_BLEURT, bleurt)

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process documents and summaries separately. """
        return summarization_processor(
            sample=sample,
            document_column=self.hyperparameters.document_column,
            summary_column=self.hyperparameters.summary_column,
            prefix=self.hyperparameters.prefix,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            additional_summaries_column=self.hyperparameters.additional_summaries_column,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument(
            '--document_column', required=False, type=str, default='document', help="Column name of document"
        )
        parser.add_argument(
            '--summary_column', required=False, type=str, default='summary', help="Column name of summary"
        )
        parser.add_argument(
            '--additional_summaries_column',
            required=False,
            type=str,
            default=None,
            help="Column name of additional summaries",
        )
        parser.add_argument(
            '--prefix', required=False, type=str, default='', help="Prefix to add before document"
        )
        parser.add_argument('--valid_bert_score', action="store_true", help="Use BERT-Score metric in validation.")
        parser.add_argument('--test_bert_score', action="store_true", help="Use BERT-Score metric in testing.")
        parser.add_argument('--valid_bleurt_score', action="store_true", help="Use BLEURT metric in validation.")
        parser.add_argument('--test_bleurt_score', action="store_true", help="Use BLEURT metric in testing.")
        add_summarization_arguments(parser)
