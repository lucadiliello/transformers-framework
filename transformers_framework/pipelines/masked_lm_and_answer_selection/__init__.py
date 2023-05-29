from transformers_framework.pipelines.masked_lm_and_answer_selection.deberta import (
    DebertaMaskedLMAndAnswerSelectionPipeline,
)
from transformers_framework.pipelines.masked_lm_and_answer_selection.deberta_v2 import (
    DebertaV2MaskedLMAndAnswerSelectionPipeline,
)
from transformers_framework.pipelines.masked_lm_and_answer_selection.roberta import (
    RobertaMaskedLMAndAnswerSelectionPipeline,
)


models = dict(
    deberta=DebertaMaskedLMAndAnswerSelectionPipeline,
    deberta_v2=DebertaV2MaskedLMAndAnswerSelectionPipeline,
    roberta=RobertaMaskedLMAndAnswerSelectionPipeline,
)
