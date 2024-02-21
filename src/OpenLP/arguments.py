import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    model_type: str = field(
        default=None,
        metadata={"help": "Name of the used model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # encoder only or not
    encoder_only: bool = field(default=False)

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    # prompting
    prompt_tuning: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use prompt or not"
        }
    )
    prompt_dim: Optional[int] = field(
        default=5,
        metadata={
            "help": "number of prompt tokens added"
        }
    )
    num_tasks: Optional[int] = field(
        default=1,
        metadata={
            "help": "number of tasks"
        }
    )
    pretrain_prompt_file: Optional[str] = field(
        default="./",
        metadata={
            "help": "pretrained prompt embedding dir"
        },
    )
    embed_init: Optional[str] = field(
        default="zero",
        metadata={
            "help": "how to initialize the prompt/prefix embedding"
        },
    )
    embed_init_file: Optional[str] = field(
        default="none",
        metadata={
            "help": "which file to load to initialize the prompt/prefix embedding"
        },
    )

    # prefix tuning
    prefix_tuning: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use prefix or not"
        }
    )
    prefix_dim: Optional[int] = field(
        default=5,
        metadata={
            "help": "number of prefix tokens added"
        }
    )
    num_experts: Optional[int] = field(
        default=20,
        metadata={
            "help": "number of expert prompts"
        }
    )

    # final projection layer
    final_proj: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use final projection layer or not"
        }
    )

    # peft config
    peft_type: Optional[str] = field(
        default="none",
        metadata={
            "help": "what type of efficient ft method"
        },
    )
    
    # mean pool final layer hidden states
    mean_pool: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use mean pooling for final representation or not"
        }
    )

    # normalize the final representation or not
    embed_normalized: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use normalization for final representation or not"
        }
    )

    # downstream prompt setting
    downstream_prompt: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use downstream prompt or not"
        }
    )
    source_agg_prompt: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use source aggregated prompt or not"
        }
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: List[str] = field(
        default=None, metadata={"help": "Path to single train file"}
    )
    eval_path: List[str] = field(
        default=None, metadata={"help": "Path to eval file"}
    )
    query_path: str = field(
        default=None, metadata={"help": "Path to query file"}
    )
    corpus_path: str = field(
        default=None, metadata={"help": "Path to corpus file"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Path to data directory"}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the single data file"}
    )
    processed_data_path: str = field(
        default=None, metadata={"help": "Path to processed data directory"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    hn_num: int = field(
        default=4, metadata={"help": "number of negatives used"}
    )
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"}
    )

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    
    encode_is_qry: bool = field(default=False)
    save_trec: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    query_column_names: str = field(
        default="id,text",
        metadata={"help": "column names for the tsv data format"}
    )
    doc_column_names: str = field(
        default="id,title,text",
        metadata={"help": "column names for the tsv data format"}
    )

    # mlm pretrain
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    mlm_probability: Optional[float] = field(
        default=0.15,
        metadata={
            "help": "The probability of token to be masked/corrupted during Mask Language Modeling"
        }
    )

    # rerank
    pos_rerank_num: int = field(default=5)
    neg_rerank_num: int = field(default=15)

    # coarse-grained node classification
    class_num: int = field(default=10)

    # upsampling or downsampling for multi-task learning
    up_sample: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Up sampling or not"
        }
    )

    def __post_init__(self):
        pass


@dataclass
class DenseTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)

    fix_lm: bool = field(default=False, metadata={"help": "fix Language Model encoder during training or not"})
    load_pretrain_prompt: bool = field(default=False, metadata={"help": "load pretrained prompt embedding or not"})

    mlm_loss: bool = field(default=False, metadata={"help": "use mlm loss or not"})
    mlm_weight: float = field(default=1, metadata={"help": "weight of mlm loss"})

    multi_weight: Optional[str] = field(default=None, metadata={"help": "Multitask objective weights"})

    l2_dist: bool = field(default=False, metadata={"help": "use l2 distance as scoring function"})

    baseline: bool = field(default=False, metadata={"help": "use baseline PLM or not"})

@dataclass
class DenseEncodingArguments(TrainingArguments):
    use_gpu: bool = field(default=False, metadata={"help": "Use GPU for encoding"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    save_path: str = field(default=None, metadata={"help": "where to save the result file"})
    retrieve_domain: str = field(default=None, metadata={"help": "name of the retrieve domain"})
    source_domain: str = field(default=None, metadata={"help": "name of the source domain"})
