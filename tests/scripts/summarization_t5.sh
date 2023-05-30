python -m transformers_framework \
    --pipeline summarization \
    --model t5 \
    \
    --precision 16 \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --devices 2 \
    \
    --pre_trained_model t5-base \
    --name t5-base-summarization-cnndailymail \
    --output_dir /tmp/transformers_framework \
    \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --train_dataset cnn_dailymail/train --train_config 3.0.0 \
    --valid_dataset cnn_dailymail/validation --valid_config 3.0.0 \
    --test_dataset cnn_dailymail/test --test_config 3.0.0 \
    --document_column article \
    --summary_column highlights \
    --prefix "summarize: " \
    --max_sequence_length 512 128 \
    \
    --learning_rate 1e-05 \
    --weight_decay 0.01 \
    --num_warmup_steps 1000 \
    --log_every_n_steps 100 \
    --monitor valid/summarization/rouge1_fmeasure \
    --early_stopping \
    --patience 5 \
    --num_workers 16 \
    \
    --generation_max_length 142 \
    --generation_min_length 56 \
    --generation_num_beams 4 \
    --generation_length_penalty 2.0 \
    --test_bert_score \
    --test_bleurt_score \
    \
    --limit_train_batches 100 --limit_val_batches 20 --limit_test_batches 20 \
    --val_check_interval 50 \
    --max_epochs 1 \
