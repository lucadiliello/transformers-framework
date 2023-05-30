python -m transformers_framework \
    --pipeline answer_selection \
    --model roberta \
    \
    --precision 16 \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --devices 2 \
    \
    --pre_trained_model roberta-base \
    --name roberta-base-extended-wikiqa \
    --output_dir /tmp/transformers_framework \
    \
    --batch_size 32 \
    --train_dataset lucadiliello/wikiqa/train \
    --valid_dataset lucadiliello/wikiqa/dev \
    --test_dataset lucadiliello/wikiqa/test \
    --input_columns question answer \
    --label_column label \
    --index_column key \
    \
    --accumulate_grad_batches 1 \
    --max_sequence_length 384 \
    --learning_rate 1e-05 \
    --log_every_n_steps 100 \
    --optimizer fuse_adam \
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
    --max_epochs 2 \
    -k 5 \
    --grouping random \
    --reload_dataloaders_every_n_epochs 1 \
    --reload_train_dataset_every_epoch \
    --prepare_data_do_not_use_cache
