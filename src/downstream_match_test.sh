DOMAIN=Geology
TEST_TASK=paper_recommendation
MAX_LEN=32

MODEL_TYPE=bert_downstream

MODEL_NAME=bert-base-uncased

PROMPT_TUNE=True
PREFIX_TUNING=False

CHECKPOINT_DIR=xxx/ckpt/$DOMAIN/downstream/$TEST_TASK/ours_load/normal/1e-3/checkpoint-4000

TEST_DIR=xxx/data/$DOMAIN/downstream/$TEST_TASK

# run test
echo "running test on downstream task: $TEST_TASK ..."
CUDA_VISIBLE_DEVICES=1 python -m OpenLP.driver.test_match  \
    --output_dir $TEST_DIR/tmp  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --tokenizer_name $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --do_eval  \
    --train_dir $TEST_DIR \
    --prompt_tuning $PROMPT_TUNE \
    --prefix_tuning $PREFIX_TUNING \
    --fp16  \
    --per_device_eval_batch_size 256 \
    --max_len $MAX_LEN  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 0
