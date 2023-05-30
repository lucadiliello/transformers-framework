python -m transformers_framework \
    --pipeline masked_lm \
    --model bert \
    \
    --precision 16 \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --devices 2 \
    \
    --pre_trained_model bert-base-cased \
    --name bert-base-cased-mlm \
    --output_dir /tmp/transformers_framework \
    \
    --batch_size 16 \
    --train_dataset lucadiliello/wikipedia_512_pretraining/train \
    --valid_dataset lucadiliello/wikipedia_512_pretraining/dev \
    --test_dataset lucadiliello/wikipedia_512_pretraining/test \
    --input_column text \
    --probability 0.15 \
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
