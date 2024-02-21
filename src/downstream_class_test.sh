DOMAIN=Mathematics
TASK_NAME=coarse_classification

TEST_DIR=xxx/data/$DOMAIN/downstream/$TASK_NAME

MODEL_TYPE=bert_downstream

# node class: sports 16, geology 18, math 17, cloth 7, home 9
CLASS_NUM=17

CHECKPOINT_DIR=xxx/ckpt/$DOMAIN/downstream/coarse_classification/ours_load/normal/1e-3/checkpoint-2600

PROMPT_TUNE=True

# run test
echo "running test..."
CUDA_VISIBLE_DEVICES=5 python -m OpenLP.driver.test_class  \
    --output_dir $TEST_DIR/tmp  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --tokenizer_name "bert-base-uncased" \
    --model_type $MODEL_TYPE \
    --do_eval  \
    --train_path $TEST_DIR/test.jsonl  \
    --eval_path $TEST_DIR/test.jsonl  \
    --class_num $CLASS_NUM \
    --prompt_tuning $PROMPT_TUNE \
    --fp16  \
    --per_device_eval_batch_size 256 \
    --max_len 32  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 32
