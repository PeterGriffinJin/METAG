DOMAIN=sports
TEST_TASK=brand
MAX_LEN=256

MODEL_TYPE=bert

MODEL_NAME=bert-base-uncased

PROMPT_TUNE=True
PREFIX_TUNING=False

CHECKPOINT_DIR=xxx/ckpt/$DOMAIN/multi_4/bert/prompt-True/prompt-dim-5/load/1,1,1,1/final-proj-False/5e-5/linear/epoch-40

TEST_DIR=xxx/data/$DOMAIN/downstream/$TEST_TASK/bert

# run test
echo "running test on downstream task: $TEST_TASK ..."
CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.test  \
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
    --dataloader_num_workers 32
