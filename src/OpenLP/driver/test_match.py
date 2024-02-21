import logging
import os
import sys

from OpenLP.arguments import DataArguments, ModelArguments
from OpenLP.arguments import DenseTrainingArguments as TrainingArguments
from OpenLP.dataset import TrainMatchCollator, TrainMatchDataset, EvalMatchDataset
from OpenLP.modeling import DenseModel
from OpenLP.trainer import DenseTrainer as Trainer
from OpenLP.utils import calculate_metrics
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.integrations import TensorBoardCallback


logger = logging.getLogger(__name__)

from IPython import embed

os.environ["WANDB_DISABLED"] = "true"

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    # this one is for bert-base ckpt
    try:
        config.prompt_tuning
    except:
        config.prompt_tuning = model_args.prompt_tuning
        config.prompt_dim = model_args.prompt_dim
        config.prefix_tuning = model_args.prefix_tuning
        config.prefix_dim = model_args.prefix_dim
        config.embed_init = model_args.embed_init
        config.num_tasks = model_args.num_tasks

    # this one is for if prefix_tuning is not in config
    try:
        config.prefix_tuning
    except:
        config.prefix_tuning = model_args.prefix_tuning
        config.prefix_dim = model_args.prefix_dim
        config.embed_init = model_args.embed_init

    try:
        config.final_proj
    except:
        config.final_proj = model_args.final_proj

    try:
        config.downstream_prompt
    except:
        config.downstream_prompt = model_args.downstream_prompt

    try:
        config.source_agg_prompt
    except:
        config.source_agg_prompt = model_args.source_agg_prompt

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    test_dataset = EvalMatchDataset(tokenizer, 'test', data_args, shuffle_seed=None, cache_dir=data_args.data_cache_dir or model_args.cache_dir)

    trainer_cls = Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        data_collator=TrainMatchCollator(
            tokenizer,
            max_len=data_args.max_len,
        ),
        compute_metrics=calculate_metrics,
    )

    metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(metrics)

if __name__ == "__main__":
    main()
