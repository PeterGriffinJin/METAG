DOMAIN=Geology

MODEL_TYPE=bert

MODEL_NAME=bert-base-uncased

PROMPT_TUNE=True
PREFIX_TUNING=False


for TEST_TASK in pp pap pvp prp pcp
do
    CHECKPOINT_DIR=xxx/ckpt/$DOMAIN/multi_5/bert-base-uncased/prompt-True/prompt-dim-5/load/1,2,2,1,1/final-proj-False/5e-5/linear/epoch-40

    TEST_DIR=xxx/data/$DOMAIN/$TEST_TASK/$MODEL_TYPE

    # run test
    echo "running test, source $TASK, target $TEST_TASK ..."
    CUDA_VISIBLE_DEVICES=1 python -m OpenLP.driver.test  \
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
        --max_len 32  \
        --evaluation_strategy steps \
        --remove_unused_columns False \
        --overwrite_output_dir True \
        --dataloader_num_workers 32
done
