python -m transformers_framework \
    --pipeline answer_selection \
    --model bert \
    \
    --precision '16-mixed' \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --devices 2 \
    \
    --pre_trained_model bert-base-cased \
    --name bert-base-cased-asnq \
    --output_dir /tmp/transformers_framework \
    \
    --batch_size 32 \
    --train_dataset lucadiliello/asnq/train \
    --valid_dataset lucadiliello/asnq/dev \
    --test_dataset lucadiliello/asnq/test \
    --input_columns question answer \
    --label_column label \
    --index_column key \
    \
    --accumulate_grad_batches 4 \
    --max_sequence_length 128 \
    --learning_rate 1e-05 \
    --log_every_n_steps 100 \
    --early_stopping \
    --patience 5 \
    \
    --weight_decay 0.001 \
    --num_warmup_steps 1000 \
    --monitor valid/answer_selection/mean_average_precision \
    --num_workers 16 \
    \
    --limit_train_batches 100 --limit_val_batches 20 --limit_test_batches 20 \
    --val_check_interval 50 \
    --max_epochs 1 \
