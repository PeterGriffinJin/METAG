DOMAIN=Mathematics
PROCESSED_DIR=xxx/data/$DOMAIN/downstream/paper_recommendation

echo "build train for matching..."
python build_train_match.py \
        --input_dir $PROCESSED_DIR \
        --output $PROCESSED_DIR \
        --tokenizer "bert-base-uncased"
