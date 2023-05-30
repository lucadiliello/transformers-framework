python -m transformers_framework \
    --pipeline masked_lm_and_seq_class \
    --model roberta \
    \
    --precision 16 \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --devices 2 \
    \
    --pre_trained_model roberta-base \
    --name roberta-base-mlm-class-quora \
    --output_dir /tmp/transformers_framework \
    \
    --batch_size 16 \
    --train_dataset HHousen/quora/train \
    --valid_dataset HHousen/quora/validation \
    --test_dataset HHousen/quora/test \
    --input_columns sentence1 sentence2 \
    --label_column label \
    \
    --accumulate_grad_batches 1 \
    --max_sequence_length 256 \
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
