python -m transformers_framework \
    --pipeline retrieval \
    --model roberta \
    \
    --precision '16-mixed' \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --devices 2 \
    \
    --pre_trained_model roberta-base \
    --name roberta-base-asnq \
    --output_dir /tmp/transformers_framework \
    \
    --batch_size 16 \
    --train_dataset lucadiliello/wikiqa/train \
    --valid_dataset lucadiliello/wikiqa/dev_clean \
    --test_dataset lucadiliello/wikiqa/dev_clean \
    --input_columns question answer \
    --label_column label \
    --index_column key \
    \
    --accumulate_grad_batches 1 \
    --max_sequence_length 64 64 \
    --learning_rate 1e-05 \
    --log_every_n_steps 100 \
    --early_stopping \
    --patience 5 \
    \
    --weight_decay 0.001 \
    --num_warmup_steps 1000 \
    --monitor valid/retrieval/mean_average_precision \
    --num_workers 12 \
    \
    --limit_train_batches 100 --limit_val_batches 20 --limit_test_batches 20 \
    --val_check_interval 50 \
    --max_epochs 1 \
