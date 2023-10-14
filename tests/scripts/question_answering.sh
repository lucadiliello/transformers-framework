python -m transformers_framework \
    --pipeline question_answering \
    --model bert \
    \
    --precision '16-mixed' \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --devices 2 \
    \
    --pre_trained_model bert-base-cased \
    --name bert-base-cased-newsqa \
    --output_dir /tmp/transformers_framework \
    \
    --batch_size 16 \
    --train_dataset lucadiliello/newsqa/train \
    --valid_dataset lucadiliello/newsqa/validation \
    --test_dataset lucadiliello/newsqa/validation \
    --query_column question \
    --context_column context \
    --answers_column answers \
    --label_column labels \
    --max_sequence_length 384 \
    --doc_stride 128 \
    --max_query_length 64 \
    \
    --accumulate_grad_batches 4 \
    --learning_rate 1e-05 \
    --log_every_n_steps 100 \
    --early_stopping \
    --patience 5 \
    \
    --weight_decay 0.001 \
    --num_warmup_steps 1000 \
    --monitor valid/question_answering/exact_match \
    --num_workers 16 \
    \
    --limit_train_batches 100 --limit_val_batches 20 --limit_test_batches 20 \
    --val_check_interval 50 \
    --max_epochs 1 \
