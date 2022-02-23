import sys

import json
import numpy as np
import logging
import argparse
import os
import sys
import time
import numpy as np
from collections import defaultdict
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import src.evaluation as evaluation
from src.cuda import CUDA
import src.data as data
import src.models as models

import pickle
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config.",
    required=True
)
parser.add_argument(
    "--checkpoint",
    help="path to model checkpoint",
    required=True
)

parser.add_argument(
    "--inference_path",
    help="path to model checkpoint",
    required=False
)

args = parser.parse_args()
config = json.load(open(args.config, 'r'))

working_dir = config['data']['working_dir']

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

config_path = os.path.join(working_dir, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='%s/train_log' % working_dir,
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info('Reading data ...')

# src, tgt = data.read_nmt_data(
#     src=config['data']['src'],
#     config=config,
#     tgt=config['data']['tgt'],
#     attribute_vocab=config['data']['attribute_vocab'],
#     ngram_attributes=config['data']['ngram_attributes']
# )

src_tgt_pickle_path = "src_tgt_pickle.pkl"
load_from_pickle_if_exists = True

if load_from_pickle_if_exists and os.path.isfile(src_tgt_pickle_path):
    src_tgt_pairs = pickle.load(open(src_tgt_pickle_path, "rb"))
    vocab_size = len(src_tgt_pairs[("tgt", 1)]['tok2id'])
    pad_idx = src_tgt_pairs[("tgt", 1)]['tok2id']['<pad>']
else:
    src_tgt_pairs = {}
    vocab_size, pad_idx = 0, 0
    for src_idx in tqdm.trange(1, 6):
        tgt_idx = (src_idx % 5+1)
        src, tgt = data.read_nmt_data(
            src=config['data'][f'dataset_{src_idx}'],
            config=config,
            tgt=config['data'][f'dataset_{src_idx}'],
            attribute_vocab=config['data']['attribute_vocab'],
            ngram_attributes=config['data']['ngram_attributes']
        )
        assert src['tok2id']['<pad>'] == tgt['tok2id']['<pad>']
        assert len(src['tok2id']) == len(tgt['tok2id'])
        vocab_size = len(src['tok2id'])
        pad_idx = src['tok2id']['<pad>']
        src_tgt_pairs[("src", src_idx)] = src
        src_tgt_pairs[("tgt", tgt_idx)] = tgt
    pickle.dump(src_tgt_pairs, open(src_tgt_pickle_path, "wb"))


test_src_tgt_pickle_path = "test_src_tgt_pickle.pkl"

if load_from_pickle_if_exists and os.path.isfile(test_src_tgt_pickle_path):
    test_src_tgt_pairs = pickle.load(open(test_src_tgt_pickle_path, "rb"))
else:
    test_src_tgt_pairs = {}
    vocab_size, pad_idx = 0, 0
    for src_idx in tqdm.trange(1, 6):
        tgt_idx = (src_idx % 5+1)
        src_test, tgt_test = data.read_nmt_data(
            src=config['data'][f'src_test_{src_idx}'],
            config=config,
            tgt=config['data'][f'tgt_test_{tgt_idx}'],
            attribute_vocab=config['data']['attribute_vocab'],
            ngram_attributes=config['data']['ngram_attributes'],
            train_src=src_tgt_pairs[("src", src_idx)],
            train_tgt=src_tgt_pairs[("tgt", tgt_idx)]
        )
        test_src_tgt_pairs[("src_test", src_idx)] = src_test
        test_src_tgt_pairs[("tgt_test", tgt_idx)] = tgt_test
    pickle.dump(test_src_tgt_pairs, open(test_src_tgt_pickle_path, "wb"))

logging.info('...done!')

src_vocab_size = vocab_size
tgt_vocab_size = vocab_size

torch.manual_seed(config['training']['random_seed'])
np.random.seed(config['training']['random_seed'])

model = models.SeqModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    pad_id_src=pad_idx,
    pad_id_tgt=pad_idx,
    config=config
)

logging.info('MODEL HAS %s params' % model.count_params())
model, start_epoch = models.attempt_load_model(model=model, checkpoint_path=args.checkpoint)
if CUDA:
    model = model.cuda()

start = time.time()
model.eval()


def inference_file(path: str):
    results = defaultdict(list)
    counts = defaultdict(int)
    sentences_count = {}
    # assume we get in every line (sentence, source, tgt)
    for line in open(path, 'r').read().split('\n')[:-1]:
        source, target = line.split(',')[-2:]
        src_idx, tgt_idx = int(source.replace(' ', '')), int(target.replace(' ', ''))
        sentence = ','.join(line.split(',')[:-2])
        if sentence[-1] == ' ':
            sentence = sentence[:-1]
        results[src_idx].append(sentence)
        sentences_count[sentence] = counts[tgt_idx]
        counts[tgt_idx] += 1

    # write files
    for src_idx in range(1, 6):
        open(config['data'][f"src_test_{src_idx}"], 'w').write('\n'.join(results[src_idx]))

    # create all combinations
    for src_idx in tqdm.trange(1, 6):
        for tgt_idx in range(1, 6):
            if src_idx == tgt_idx:
                continue
            bleu, edit_distance, inputs, preds, golds, auxs = evaluation.inference_metrics(
                model, test_src_tgt_pairs[("src_test", src_idx)], src_tgt_pairs[("tgt", tgt_idx)], config)
            open(working_dir + f'/preds_{src_idx}to{tgt_idx}', 'w').write('\n'.join(preds) + '\n')

    # find the right order of the sentences
    final = []
    for line in open(path, 'r').read().split('\n'):
        source, target = line.split(',')[-2:]
        src_idx, tgt_idx = int(source.replace(' ', '')), int(target.replace(' ', ''))
        results_path = working_dir + f'/preds_{src_idx}to{tgt_idx}'
        sentence = ','.join(line.split(',')[:-2])
        if sentence[-1] == ' ':
            sentence = sentence[:-1]
        final.append(open(results_path, 'r').read().split('\n')[sentences_count[sentence]])
    print('\n'.join(final))

if hasattr(args, 'inference_path'):
    inference_file(args.inference_path)
