python -m transformers_framework \
    --pipeline denoising \
    --model bart \
    \
    --precision 16 \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --devices 2 \
    \
    --pre_trained_model facebook/bart-base \
    --name bart-base-denoised \
    --output_dir /tmp/transformers_framework \
    \
    --batch_size 8 \
    --train_dataset lucadiliello/wikipedia_512_pretraining/train \
    --valid_dataset lucadiliello/wikipedia_512_pretraining/dev \
    --test_dataset lucadiliello/wikipedia_512_pretraining/test \
    --input_column text \
    --max_sequence_length 128 \
    \
    --probability 0.3 \
    --max_number_of_spans 200 \
    --whole_word_denoising \
    --shuffle_sentences \
    --mean_span_length 2.5 \
    \
    --log_every_n_steps 1000 \
    --accumulate_grad_batches 8 \
    --optimizer adamw \
    --scheduler linear_decay \
    --learning_rate 1e-04 \
    --weight_decay 0.01 \
    --num_warmup_steps 1000 \
    --num_workers 16 \
    --num_sanity_val_steps 0 \
    \
    --limit_train_batches 100 --limit_val_batches 20 --limit_test_batches 20 \
    --val_check_interval 50 \
    --max_epochs 1 \
