LOSS = 'loss'

# masked language modeling
MASKED_LM_LOSS = 'masked_lm/loss'
MASKED_LM_ACCURACY = 'masked_lm/accuracy'
MASKED_LM_F1 = 'masked_lm/f1'
MASKED_LM_PERPLEXITY = 'masked_lm/perplexity'

# sequence classification
SEQ_CLASS_LOSS = 'seq_class/loss'
SEQ_CLASS_ACCURACY = 'seq_class/accuracy'
SEQ_CLASS_F1 = 'seq_class/f1'

# answer selection
ANSWER_SELECTION_MAP = 'answer_selection/mean_average_precision'
ANSWER_SELECTION_MRR = 'answer_selection/mean_reciprocal_rank'
ANSWER_SELECTION_PRECISION = lambda i: f'answer_selection/precision@{i}'  # noqa: E731
ANSWER_SELECTION_NDCG = 'answer_selection/ndgc'
ANSWER_SELECTION_HR = lambda i: f'answer_selection/hitrate@{i}'  # noqa: E731

# information retrieval
RETRIEVAL_LOSS = 'retrieval/loss'
RETRIEVAL_ACCURACY = 'retrieval/accuracy'
RETRIEVAL_F1 = 'retrieval/f1'
RETRIEVAL_MAP = 'retrieval/mean_average_precision'
RETRIEVAL_MRR = 'retrieval/mean_reciprocal_rank'
RETRIEVAL_PRECISION = lambda i: f'retrieval/precision@{i}'  # noqa: E731
RETRIEVAL_NDCG = 'retrieval/ndgc'
RETRIEVAL_HR = lambda i: f'retrieval/hitrate@{i}'  # noqa: E731

# seq-to-seq language modeling
SEQ_TO_SEQ_LM_LOSS = 'seq_to_seq_lm/loss'
SEQ_TO_SEQ_LM_ACCURACY = 'seq_to_seq_lm/accuracy'
SEQ_TO_SEQ_LM_F1 = 'seq_to_seq_lm/f1'

# machine reading (extractive QA)
QUESTION_ANSWERING_LOSS = "question_answering/loss"
QUESTION_ANSWERING_START_ACCURACY = "question_answering/start_accuracy"
QUESTION_ANSWERING_END_ACCURACY = "question_answering/end_accuracy"
QUESTION_ANSWERING_EXACT_MATCH = "question_answering/exact_match"
QUESTION_ANSWERING_F1 = "question_answering/f1"

# summarization
SUMMARIZATION_BLEU = "summarization/bleu"
SUMMARIZATION_BLEURT = "summarization/bleurt"
SUMMARIZATION_BERTSCORE = "summarization/bert_score"
SUMMARIZATION_ROUGE = "summarization"

# token classification
TOKEN_CLASS_LOSS = "token_classification/loss"
TOKEN_CLASS_ACCURACY = "token_classification/accuracy"
TOKEN_CLASS_F1 = "token_classification/f1"
TOKEN_CLASS_PERPLEXITY = "token_classification/perplexity"

# token detection
TOKEN_DETECTION_LOSS = "token_detection/loss"
TOKEN_DETECTION_ACCURACY = "token_detection/accuracy"
TOKEN_DETECTION_F1 = "token_detection/f1"
TOKEN_DETECTION_PERPLEXITY = "token_detection/perplexity"
