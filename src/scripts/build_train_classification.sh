DOMAIN=home
PROCESSED_DIR=xxx/data/$DOMAIN/downstream/coarse_classification

echo "build train for coarse-grained classification..."
python build_train_ncc.py \
        --input_dir $PROCESSED_DIR \
        --output $PROCESSED_DIR \
        --tokenizer "bert-base-uncased"
