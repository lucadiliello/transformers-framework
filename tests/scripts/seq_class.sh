python -m transformers_framework \
    --pipeline seq_class \
    --model bert \
    \
    --precision '16-mixed' \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --devices 2 \
    \
    --pre_trained_model bert-base-cased \
    --name bert-base-cased-quora \
    --output_dir /tmp/transformers_framework \
    \
    --batch_size 64 \
    --train_dataset HHousen/quora/train \
    --valid_dataset HHousen/quora/validation \
    --test_dataset HHousen/quora/test \
    --input_columns sentence1 sentence2 \
    --label_column label \
    \
    --accumulate_grad_batches 1 \
    --max_sequence_length 128 \
    --learning_rate 1e-05 \
    --log_every_n_steps 100 \
    --early_stopping \
    --patience 5 \
    \
    --weight_decay 0.001 \
    --num_warmup_steps 1000 \
    --monitor valid/seq_class/accuracy \
    --num_workers 16 \
    \
    --limit_train_batches 100 --limit_val_batches 20 --limit_test_batches 20 \
    --val_check_interval 50 \
    --max_epochs 1 \
