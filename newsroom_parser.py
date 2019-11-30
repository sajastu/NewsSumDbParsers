import dbm
import json
import os
import pickle
import sys
from tqdm import tqdm
import subprocess

import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def read_news(newsroom_dir, set):
    out = []
    instance = {}

    total_cnt = file_len(newsroom_dir + '/' + set + '.jsonl')
    with open(newsroom_dir + '/' + set + '.jsonl', mode='r') as file:
        for line in tqdm(file, total=total_cnt):
            news = json.loads(line)
            instance['text'] = news['text']
            instance['summary'] = news['summary']
            out.append(instance.copy())
            instance.clear()
    return out


def preprocess_tokenize(collection, db_dir):

    out = []
    tmp = {}

    for instance in tqdm(collection, total=len(collection)):
        src = instance['text'].replace('\n', ' ').replace('\r', ' ')
        tgt = instance['summary'].replace('\n', ' ').replace('\r', ' ')

        # doc_src = CachedNlp(db_dir).tokenize(src, ['parser', 'tagger'])
        # doc_tgt = CachedNlp(db_dir).tokenize(tgt, ['parser', 'tagger'])
        doc_src = nlp(src, disable=['parser', 'tagger'])
        doc_tgt = nlp(tgt, disable=['parser', 'tagger'])

        src_arr = []
        tgt_arr = []
        for token_s in doc_src:
            if len(token_s.text.strip()) > 0:
                src_arr.append(token_s.text)
        for token_tgt in doc_tgt:
            if len(token_tgt.text.strip()) > 0:
                tgt_arr.append(token_tgt.text)

        tmp['text'] = src_arr.copy()
        tmp['summary'] = tgt_arr.copy()
        src_arr.clear()
        tgt_arr.clear()
        out.append(tmp.copy())

    return out


def write_jsonl(collection, set_name):

    if not os.path.exists(newsroom_dir + '/tokenized/'):
        os.mkdir(newsroom_dir + '/tokenized/')

    out_file = open(newsroom_dir + '/tokenized/' + set_name + '.jsonl', mode='a')

    for instance in collection:
        json.dump(instance, out_file)
        out_file.write('\n')


class CachedNlp(object):
    def __init__(self, dir):
        self.cache_file = dbm.open(dir, 'c')

    def tokenize(self, *args):
        key = repr(args[0])
        if key in self.cache_file:
            return pickle.loads(self.cache_file[key])
        result = nlp(args[0], disable=args[1])
        tokens = [r.text for r in result]
        self.cache_file[key] = pickle.dumps(tokens)
        return tokens


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python newsroom_parser.py <newsroom root dir>")
        sys.exit()
    newsroom_dir = sys.argv[1]

    # Reading news from json file into lists
    print('Reading files...')
    train = read_news(newsroom_dir, 'train')
    dev = read_news(newsroom_dir, 'dev')
    test = read_news(newsroom_dir, 'test')

    # Doing some sort of pre-processing to remove unnecessary symbols, and then tokenizing...
    print('Preprocessing and then tokenizing...')

    train = preprocess_tokenize(train, newsroom_dir)
    print('Dumping traon to jsonl...')
    write_jsonl(train, 'train')
    train.clear()

    dev = preprocess_tokenize(dev, newsroom_dir)
    print('Dumping dev to jsonl...')
    write_jsonl(dev, 'dev')
    dev.clear()

    test = preprocess_tokenize(test, newsroom_dir)
    print('Dumping test to jsonl...')
    write_jsonl(test, 'test')
    test.clear()

    print('Done!!')
