DOMAIN=cloth
TASK_NAME=price

TEST_DIR=xxx/data/$DOMAIN/downstream/$TASK_NAME

MODEL_TYPE=bert_downstream

CHECKPOINT_DIR=xxx/ckpt/$DOMAIN/downstream/$TASK_NAME/ours_load/normal/1e-2/checkpoint-2000

PROMPT_TUNE=True

# run test
echo "running test..."
CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.test_regression  \
    --output_dir $TEST_DIR/tmp  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --tokenizer_name "bert-base-uncased" \
    --model_type $MODEL_TYPE \
    --do_eval  \
    --train_path $TEST_DIR/test.jsonl  \
    --eval_path $TEST_DIR/test.jsonl  \
    --prompt_tuning $PROMPT_TUNE \
    --prompt_dim 5 \
    --fp16  \
    --downstream_prompt $DOWNSTREAM_PROMPT \
    --source_agg_prompt $SOURCE_AGG_PROMPT \
    --per_device_eval_batch_size 256 \
    --max_len 32  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 8
