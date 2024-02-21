DOMAIN=sports
NUM_TASKS=4
MODE=multi_$NUM_TASKS

LR="5e-5"
EMBED_INIT="load"
LR_scheduler_type="linear"
MODEL_TYPE=bert

MODEL_NAME=bert-base-uncased

PROMPT_TUNE=True
FINAL_PROJ=False
PROMPT_DIM=5
EPOCH_NUM=40
MULTI_WEIGHT="1,1,1,1"

PROCESSED_DIR=xxx/data/$DOMAIN/$MODE/$MODEL_TYPE
LOG_DIR=xxx/logs/$DOMAIN/$MODE
CHECKPOINT_DIR=xxx/ckpt/$DOMAIN/$MODE


export CUDA_VISIBLE_DEVICES=0,1,2,5

echo "start training..."

python -m torch.distributed.launch --nproc_per_node=4 --master_port 19288 \
    -m OpenLP.driver.train_multitask  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/prompt-$PROMPT_TUNE/prompt-dim-$PROMPT_DIM/$EMBED_INIT/$MULTI_WEIGHT/final-proj-$FINAL_PROJ/$LR/$LR_scheduler_type/epoch-$EPOCH_NUM  \
    --model_name_or_path $MODEL_NAME  \
    --model_type $MODEL_TYPE \
    --do_train  \
    --save_steps 500  \
    --eval_steps 500  \
    --logging_steps 10 \
    --train_dir $PROCESSED_DIR \
    --prompt_tuning $PROMPT_TUNE \
    --embed_init $EMBED_INIT \
    --embed_init_file $PROCESSED_DIR/hard_prompt_embed2.pt \
    --num_tasks $NUM_TASKS \
    --multi_weight $MULTI_WEIGHT \
    --fp16  \
    --prompt_dim $PROMPT_DIM \
    --final_proj $FINAL_PROJ \
    --per_device_train_batch_size 128  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --lr_scheduler_type $LR_scheduler_type \
    --max_len 32  \
    --num_train_epochs $EPOCH_NUM  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/prompt-$PROMPT_TUNE/prompt-dim-$PROMPT_DIM/$EMBED_INIT/$MULTI_WEIGHT/final-proj-$FINAL_PROJ/$LR/$LR_scheduler_type/epoch-$EPOCH_NUM  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --dataloader_num_workers 16
