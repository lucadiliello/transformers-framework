# Pipelines

Pipelines define the processing of data and training setup for different techniques.

Available pipelines, in round brackets the nickname to use in the launching script.

- Answer Selection (`answer_selection`)
- Sequence Classification (`seq_class`)
- Denoising BART/T5 (`denoising`)
- Extractive Question Answering (`question_answering`)
- Masked Language Modeling (`masked_lm`)
  - \+ Answer Selection (`masked_lm_and_answer_selection`)
  - \+ Sequence Classification (`masked_lm_and_seq_class`)
  - \+ Token Classification (`masked_lm_and_token_class`)
- Summarization (`summarization`)
- Token Classification (`token_class`)
- POS Tagging (`pos_tagging`)
- Named Entity Recognition (`named_entity_recognition`)
- Token Detection (`random_token_detection`)
  - \+ Masked Language Modeling (`token_detection_masked_lm_models`)
    - \+ Classification (`token_detection_masked_lm_seq_class_models`)
