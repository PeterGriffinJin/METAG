# This code script is for multitask learning with in batch negative samples.

import logging
import os
import sys
import glob

from OpenLP.arguments import DataArguments, ModelArguments
from OpenLP.arguments import DenseTrainingArguments as TrainingArguments
from OpenLP.dataset import TrainCollator, TrainSingleDataset, EvalSingleDataset
from OpenLP.modeling import DenseMultiModel
from OpenLP.trainer import MultiDenseTrainer as Trainer
from OpenLP.utils import calculate_multitask_metrics
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.integrations import TensorBoardCallback


logger = logging.getLogger(__name__)

from IPython import embed

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
    assert not (model_args.prompt_tuning and model_args.prefix_tuning)
    config.prompt_tuning = model_args.prompt_tuning
    config.prompt_dim = model_args.prompt_dim
    config.prefix_tuning = model_args.prefix_tuning
    config.prefix_dim = model_args.prefix_dim
    config.embed_init = model_args.embed_init
    config.embed_init_file = model_args.embed_init_file
    config.final_proj = model_args.final_proj
    config.num_tasks = model_args.num_tasks
    config.num_experts = model_args.num_experts
    if data_args.train_path:
        assert model_args.num_tasks == len(data_args.train_path)
    if data_args.eval_path:
        assert model_args.num_tasks == len(data_args.eval_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DenseMultiModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    assert model_args.model_type in ['bert', 't5', 'bert_moe']

    # load pretrained prompt or not
    if training_args.load_pretrain_prompt:
        model.lm.load_prompt_weights(model_args.pretrain_prompt_file)

    # fix parameters or not
    if training_args.fix_lm:
        assert model_args.prompt_tuning == True or model_args.prefix_tuning == True
        for param in model.lm.parameters():
            param.requires_grad = False
        model.lm.prompt_emb.requires_grad = True
        model.lm.prefix_emb.requires_grad = True

    # read in datasets
    train_files = glob.glob(os.path.join(data_args.train_dir, "train.*.jsonl"))
    eval_files = glob.glob(os.path.join(data_args.train_dir, "val.*.jsonl"))

    train_dataset = [TrainSingleDataset(train_file, tokenizer, data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
                    for train_file in train_files]
    eval_dataset = [EvalSingleDataset(eval_file, tokenizer, 'val', data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
                    for eval_file in eval_files]

    # load task id dict
    for train_d in train_dataset:
        train_d.load_task_n2id(data_args.train_dir)
    for eval_d in eval_dataset:
        eval_d.load_task_n2id(data_args.train_dir)

    assert len(train_dataset[0].task_n2id) == model_args.num_tasks
    assert len(eval_dataset[0].task_n2id) == model_args.num_tasks

    # make sure the task id order (increase) is correct
    assert len(set([train_d[0]['task_id'] for train_d in train_dataset])) == model_args.num_tasks
    assert len(set([eval_d[0]['task_id'] for eval_d in eval_dataset])) == model_args.num_tasks
    train_dataset.sort(key=lambda x: x[0]['task_id'])
    eval_dataset.sort(key=lambda x: x[0]['task_id'])

    # multitask weight
    if training_args.multi_weight:
        training_args.multi_weight = [float(n) for n in training_args.multi_weight.split(',')]
        assert len(training_args.multi_weight) == model_args.num_tasks
    else:
        training_args.multi_weight = [1.0 for _ in range(model_args.num_tasks)]

    tb_callback = TensorBoardCallback()
    metrics_calculator = calculate_multitask_metrics(data_args.train_dir)

    trainer_cls = Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TrainCollator(
            tokenizer,
            max_len=data_args.max_len,
        ),
        callbacks=[tb_callback],
        compute_metrics=metrics_calculator,
    )
    for train_d in train_dataset:
        train_d.trainer = trainer
    trainer.up_sample = data_args.up_sample
    trainer.num_tasks = model_args.num_tasks 
    trainer.task_names = [name for (name, idd) in sorted([(name, train_dataset[0].task_n2id[name]) for name in train_dataset[0].task_n2id], key=lambda x: x[1])]

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
