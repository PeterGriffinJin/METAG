import os
import json
import random
from argparse import ArgumentParser

from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--prefix', type=str, required=True)

args = parser.parse_args()


for set_n in ['train', 'val', 'test']:
    with open(f'{args.data_dir}/{set_n}.{args.prefix}.jsonl') as f:
        readin = f.readlines()
    with open(f'{args.data_dir}/{set_n}.{args.prefix}.jsonl', 'w') as fout:
        for line in tqdm(readin):
            tmp = json.loads(line)
            fout.write(json.dumps({'q_text':tmp['q_text'], 'k_text':tmp['k_text'], 'task': args.prefix})+'\n')
