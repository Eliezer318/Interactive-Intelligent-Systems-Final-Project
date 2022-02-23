import pandas as pd
import string
from tqdm import tqdm
from collections import defaultdict
import re
from langdetect import detect
PATH = '/home/student/yelp_academic_dataset_review.json'
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def preprocess(sentence: str):
    is_numeric = lambda word: '_num_' if word.isnumeric() or (word[1:].isnumeric() and word[0] == '-') else word
    sentence = re.sub('\s{2,}', ' ', re.sub('([.,!?()])', r' \1 ', sentence))
    words = [is_numeric(word) for word in sentence.lower().split()]
    return ' '.join(words)


def split_datasets():
    stars_list = defaultdict(list)
    chunks = pd.read_json(PATH, chunksize=100_000, lines=True)
    for chunk in tqdm(chunks, desc='Mining sentences from yelp dataset', total=87):
        chunk_shorts = chunk.loc[chunk.text.str.split().str.len() < 15].copy()
        chunk_shorts['text'] = chunk_shorts.text.apply(preprocess)
        for i in range(1, 6):
            texts = chunk_shorts.loc[chunk_shorts.stars == i].text.tolist()
            texts = [text for text in texts if len(set(text)) > 4 and detect(text) == 'en']
            with open(f'/home/student/Code/data/new/sentiment.train.{i}', 'a') as f:
                f.writelines('\n'.join(texts) + '\n')


def pick_sentences():
    for source in range(1, 6):
        open(f'../data/new/sentiment.test.{source}', 'w').write('\n'.join(open(f'../data/new/sentiment.train.{source}', 'r').read().split('\n')[:50]))


def organize_results():
    N = 27
    for source_idx in range(1, 6):
        results = pd.DataFrame(columns=[source_idx] + [target_idx for target_idx in range(1, 6) if target_idx != source_idx])
        # input_sentences = open(input_path, 'r').read().split('\n')[:10]
        target_results = []
        for target_idx in range(1, 6):
            if source_idx == target_idx:
                input_path = f'../data/new/sentiment.test.{source_idx}'
                target_results.append(open(input_path, 'r').read().split('\n')[:50])
                continue
            output_i_path = f'../working_dir/preds_{source_idx}to{target_idx}'
            target_results.append(open(output_i_path, 'r').read().split('\n'))
        for a in list(zip(*target_results))[N:N + 3]:
            print(f'source is {source_idx}')
            for i, b in enumerate(a, start=1):
                print(i, b)
            print('\n\n\n')


split_datasets()
pick_sentences()
# organize_results()

