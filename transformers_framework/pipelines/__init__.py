from transformers_framework.pipelines.answer_selection import models as answer_selection_models
from transformers_framework.pipelines.cluster_random_token_detection import \
    models as cluster_random_token_detection_models
from transformers_framework.pipelines.denoising import models as denoising_models
from transformers_framework.pipelines.masked_lm import models as masked_lm_models
from transformers_framework.pipelines.masked_lm_and_answer_selection import models as masked_lm_answer_selection_models
from transformers_framework.pipelines.masked_lm_and_seq_class import models as masked_lm_seq_class_models
from transformers_framework.pipelines.masked_lm_and_token_class import models as masked_lm_token_class_models
from transformers_framework.pipelines.question_answering import models as question_answering_models
from transformers_framework.pipelines.random_token_detection import models as random_token_detection_models
from transformers_framework.pipelines.seq_class import models as seq_class_models
from transformers_framework.pipelines.summarization import models as summarization_models
from transformers_framework.pipelines.token_class import models as token_class_models
from transformers_framework.pipelines.token_detection_and_masked_lm import models as token_detection_masked_lm_models
from transformers_framework.pipelines.token_detection_and_masked_lm_and_seq_class import \
    models as token_detection_masked_lm_seq_class_models


pipelines = dict(
    answer_selection=answer_selection_models,
    seq_class=seq_class_models,
    masked_lm=masked_lm_models,
    masked_lm_and_answer_selection=masked_lm_answer_selection_models,
    masked_lm_and_seq_class=masked_lm_seq_class_models,
    masked_lm_and_token_class=masked_lm_token_class_models,
    token_class=token_class_models,
    pos_tagging=token_class_models,
    named_entity_recognition=token_class_models,
    random_token_detection=random_token_detection_models,
    cluster_random_token_detection=cluster_random_token_detection_models,
    token_detection_and_masked_lm=token_detection_masked_lm_models,
    token_detection_and_masked_lm_and_seq_class=token_detection_masked_lm_seq_class_models,
    question_answering=question_answering_models,
    denoising=denoising_models,
    summarization=summarization_models,
)
