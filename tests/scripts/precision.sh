# test all precisions
for precision in '16-mixed' '16-true' 'bf16-mixed' 'bf16-true' '32-true' '64-true' 'nf4' 'f4-dq' 'fp4' 'fp4-dq' 'int8' 'int8-training' "transformer-engine" "transformer-engine-float16"; do

    python -m transformers_framework \
        --pipeline answer_selection \
        --model roberta \
        \
        --precision $precision \
        --accelerator gpu --strategy ddp \
        --devices 2 \
        \
        --pre_trained_model roberta-base \
        --name roberta-base-wikiqa-precision-$precision \
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
        --max_sequence_length 128 \
        --learning_rate 1e-05 \
        --log_every_n_steps 100 \
        --optimizer adamw \
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
        --max_epochs 2
done
