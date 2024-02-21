# # Academic
# DOMAIN=Mathematics
# DATA_DIR=xxx/data/$DOMAIN

# PLM=bert
# TOKENIZER='bert-base-uncased'


# for PREFIX in pp pap pvp prp pcp
# do
#     mkdir $DATA_DIR/$PREFIX
#     mkdir $DATA_DIR/$PREFIX/text
#     mkdir $DATA_DIR/$PREFIX/$PLM
#     echo "split data for $PREFIX"
#     python split.py --input_data $DATA_DIR/raw/$PREFIX.jsonl --output_dir $DATA_DIR/$PREFIX/text --train_size 1000000 --val_size 100000 --test_size 100000 --prefix $PREFIX
#     echo "tokenizing data for $PREFIX"
#     python build_train.py --input_dir $DATA_DIR/$PREFIX/text --output $DATA_DIR/$PREFIX/$PLM --tokenizer $TOKENIZER --prefix $PREFIX
#     echo "adding task id for $PREFIX"
#     python add_taskid.py --data_dir $DATA_DIR --plm $PLM --prefix $PREFIX
# done


# # Amazon
DOMAIN=sports
DATA_DIR=xxx/data/$DOMAIN

PLM=bert
TOKENIZER='bert-base-uncased'


for PREFIX in also_bought also_viewed cobrand
do
    mkdir $DATA_DIR/$PREFIX
    mkdir $DATA_DIR/$PREFIX/text
    mkdir $DATA_DIR/$PREFIX/$PLM
    echo "split data for $PREFIX"
    python split.py --input_data $DATA_DIR/raw/$PREFIX.jsonl --output_dir $DATA_DIR/$PREFIX/text --train_size 100000 --val_size 10000 --test_size 10000 --prefix $PREFIX
    echo "tokenizing data for $PREFIX"
    python build_train.py --input_dir $DATA_DIR/$PREFIX/text --output $DATA_DIR/$PREFIX/$PLM --tokenizer $TOKENIZER --prefix $PREFIX
    echo "adding task id for $PREFIX"
    python add_taskid.py --data_dir $DATA_DIR --plm $PLM --prefix $PREFIX
done


for PREFIX in bought_together
do
    mkdir $DATA_DIR/$PREFIX
    mkdir $DATA_DIR/$PREFIX/text
    mkdir $DATA_DIR/$PREFIX/$PLM
    echo "split data for $PREFIX"
    python split.py --input_data $DATA_DIR/raw/$PREFIX.jsonl --output_dir $DATA_DIR/$PREFIX/text --train_size 50000 --val_size 5000 --test_size 5000 --prefix $PREFIX
    echo "tokenizing data for $PREFIX"
    python build_train.py --input_dir $DATA_DIR/$PREFIX/text --output $DATA_DIR/$PREFIX/$PLM --tokenizer $TOKENIZER --prefix $PREFIX
    echo "adding task id for $PREFIX"
    python add_taskid.py --data_dir $DATA_DIR --plm $PLM --prefix $PREFIX
done
