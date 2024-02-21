DOMAIN=Mathematics
PROCESSED_DIR=xxx/data/$DOMAIN/downstream/year_prediction

echo "build train for regression..."
python build_train_ncc.py \
        --input_dir $PROCESSED_DIR \
        --output $PROCESSED_DIR \
        --tokenizer "bert-base-uncased"
