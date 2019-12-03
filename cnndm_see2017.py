""" This code is partially uses functions implemented by See et al."""

import hashlib
import json
import os
import sys

import spacy
from tqdm import tqdm

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"


def read_txt_file(dir):
    lines = []
    with open(dir, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


def extract_sections(story_file):
    lines = read_txt_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join(highlights)

    return article, abstract


def preprocess_tokenize(file_names, cnn_dir, dm_dir):
    out = []
    tmp = {}
    for f_name in tqdm(file_names, total=len(file_names)):
        story_file = ''
        # Look in the tokenized story dirs to find the .story file corresponding to this url
        if os.path.isfile(os.path.join(cnn_dir, f_name + '.story')):
            story_file = os.path.join(cnn_dir, f_name + '.story')
        elif os.path.isfile(os.path.join(dm_dir, f_name + '.story')):
            story_file = os.path.join(dm_dir, f_name + '.story')
        else:
            print("Error: Couldn't find story file %s" % story_file)

        if len(story_file) > 0:
            article, abstract = extract_sections(story_file)

            doc_src = nlp(article, disable=['parser', 'tagger'])
            doc_tgt = nlp(abstract, disable=['parser', 'tagger'])
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
        else:
            print('Did not added')

    return out


def write_jsonl(data_arr, set_name):
    if not os.path.exists('/disk1/sajad/datasets/cnndm/tokenized/'):
        os.mkdir('/disk1/sajad/datasets/cnndm/tokenized/')

    out_file = open('/disk1/sajad/datasets/cnndm/tokenized/' + set_name + '.jsonl', mode='a')

    for instance in tqdm(data_arr, total=len(data_arr)):
        json.dump(instance, out_file)
        out_file.write('\n')

# Sample command: python cnndm_see2017.py /disk1/sajad/datasets/cnndm/cnn_stories_tokenized/ /disk1/sajad/datasets/cnndm/dm_stories_tokenized/ &> c.l &
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python cnndm_see2017.py <cnn_stories_dir> <dailymail_stories_dir>")
        sys.exit()
    cnn_stories_dir = sys.argv[1]
    dm_stories_dir = sys.argv[2]

    train_fnames = get_url_hashes(read_txt_file(all_train_urls))
    dev_fnames = get_url_hashes(read_txt_file(all_val_urls))
    test_fnames = get_url_hashes(read_txt_file(all_test_urls))

    train_data = preprocess_tokenize(train_fnames, cnn_stories_dir, dm_stories_dir)
    write_jsonl(train_data, 'train')
    train_data.clear()

    dev_data = preprocess_tokenize(dev_fnames, cnn_stories_dir, dm_stories_dir)
    write_jsonl(dev_data, 'dev')
    dev_data.clear()

    test_data = preprocess_tokenize(test_fnames, cnn_stories_dir, dm_stories_dir)
    write_jsonl(test_data, 'test')
    test_data.clear()
