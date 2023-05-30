python -m transformers_framework \
    --pipeline masked_lm_and_token_class \
    --model roberta \
    \
    --precision 16 \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --devices 2 \
    \
    --pre_trained_model roberta-base \
    --name roberta-base-mlm-token-class-conll2003 \
    --output_dir /tmp/transformers_framework \
    \
    --batch_size 16 \
    --train_dataset conll2003/train \
    --valid_dataset conll2003/validation \
    --test_dataset conll2003/test \
    --input_column tokens \
    --label_column ner_tags \
    --num_labels 10 \
    --probability 0.15 \
    \
    --accumulate_grad_batches 1 \
    --max_sequence_length 128 \
    --learning_rate 1e-05 \
    --log_every_n_steps 100 \
    \
    --weight_decay 0.001 \
    --num_warmup_steps 1000 \
    --num_workers 16 \
    \
    --limit_train_batches 100 --limit_val_batches 20 --limit_test_batches 20 \
    --val_check_interval 50 \
    --max_epochs 1 \
