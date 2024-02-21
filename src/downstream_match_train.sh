DOMAIN=Mathematics
NUM_TASKS=5

TASK_NAME=paper_recommendation

PROCESSED_DIR=xxx/data/$DOMAIN/downstream/$TASK_NAME
LOG_DIR=xxx/logs/$DOMAIN/downstream/$TASK_NAME
CHECKPOINT_DIR=xxx/ckpt/$DOMAIN/downstream/$TASK_NAME

LR="1e-3"
MODEL_TYPE=bert_downstream

MODEL_DIR=xxx/ckpt/$DOMAIN/multi_$NUM_TASKS/bert-base-uncased/prompt-False/prompt-dim-5/zero/1,1,1,1,1/final-proj-True/5e-5/linear/epoch-40

MODEL_SAVE_NAME="ours_load"

echo "start training..."

EMBED_INIT=normal
PROMPT_TUNE=True
FIX_LM=True
PROMPT_DIM=5

FINAL_PROJ=False
DOWNSTREAM_PROMPT=False
SOURCE_AGG_PROMPT=True

# bert-base
CUDA_VISIBLE_DEVICES=7 python -m OpenLP.driver.train_match  \
    --output_dir $CHECKPOINT_DIR/$MODEL_SAVE_NAME/$EMBED_INIT/$LR  \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name 'bert-base-uncased' \
    --model_type $MODEL_TYPE \
    --do_train  \
    --save_steps 1000  \
    --eval_steps 1000  \
    --logging_steps 500 \
    --train_path $PROCESSED_DIR/train.jsonl  \
    --eval_path $PROCESSED_DIR/val.jsonl  \
    --num_tasks $NUM_TASKS \
    --prompt_tuning $PROMPT_TUNE \
    --final_proj $FINAL_PROJ \
    --fix_lm $FIX_LM \
    --fp16  \
    --downstream_prompt $DOWNSTREAM_PROMPT \
    --source_agg_prompt $SOURCE_AGG_PROMPT \
    --prompt_dim $PROMPT_DIM \
    --embed_init $EMBED_INIT \
    --per_device_train_batch_size 128  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --max_len 32  \
    --num_train_epochs 500  \
    --logging_dir $LOG_DIR/$MODEL_SAVE_NAME/$EMBED_INIT/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --seed 42 \
    --dataloader_num_workers 0
