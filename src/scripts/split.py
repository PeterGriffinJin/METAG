import os
import json
import random
from argparse import ArgumentParser

from tqdm import tqdm

random.seed(42)

parser = ArgumentParser()
parser.add_argument('--input_data', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--train_size', type=int, required=True)
parser.add_argument('--val_size', type=int, required=True)
parser.add_argument('--test_size', type=int, required=True)
parser.add_argument('--prefix', type=str, required=True)

args = parser.parse_args()


with open(args.input_data) as f:
    readin = f.readlines()
    random.shuffle(readin)
    with open(os.path.join(args.output_dir, f'train.{args.prefix}.text.jsonl'), 'w') as fout:
        for line in readin[:args.train_size]:
            fout.write(line)
    with open(os.path.join(args.output_dir, f'val.{args.prefix}.text.jsonl'), 'w') as fout:
        for line in readin[args.train_size:(args.train_size + args.val_size)]:
            fout.write(line)
    with open(os.path.join(args.output_dir, f'test.{args.prefix}.text.jsonl'), 'w') as fout:
        for line in readin[(args.train_size + args.val_size) : (args.train_size + args.val_size + args.test_size)]:
            fout.write(line)
