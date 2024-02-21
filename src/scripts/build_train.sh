PREFIX=pvp
PROCESSED_DIR=xxx/data/CS/$PREFIX/text

echo "build train for link prediction ($PREFIX)..."
python build_train.py \
        --input_dir $PROCESSED_DIR \
        --output $PROCESSED_DIR \
        --tokenizer 't5-base' \
        --prefix $PREFIX
