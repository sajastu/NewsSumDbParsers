import json
import sys
import spacy

nlp = spacy.load("en_core_web_sm")

def read_news(newsroom_dir, set):
    out = []
    instance = {}
    with open(newsroom_dir + set + '.jsonl', mode='r') as file:
        for line in file:
            news = json.loads(line)
            instance['text'] = news['text']
            instance['summary'] = news['summary']
            import pdb;pdb.set_trace()
            out.append(instance.copy())
            instance.clear()
    return out


def preprocess_tokenize(collection):
    out = []
    tmp = {}
    for instance in collection:
        src = instance['text'].replace('\n', ' ').replace('\r', ' ')
        tgt = instance['summary'].replace('\n', ' ').replace('\r', ' ')

        doc_src = nlp(src, disable=['parser'])
        doc_tgt = nlp(tgt, disable=['parser'])
        src_arr = []
        tgt_arr = []
        for token_s in doc_src:
            src_arr.append(token_s.text)
        for token_tgt in doc_tgt:
            src_arr.append(token_tgt.text)

        tmp['text'] = src_arr.copy()
        tmp['summary'] = tgt_arr.copy()

        out.append(tmp)

    return out


def write_jsonl(collection, set_name):
    out_file = open('./' + newsroom_dir + 'tokenized/' + set_name + '.jsonl', mode='a')

    for instance in collection:
        json.dump(instance, out_file)
        json.dump('\n')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python newsroom_parser.py <newsroom root dir>")
        sys.exit()
    newsroom_dir = sys.argv[1]
    br = 0

    # Reading news from json file into lists
    train = read_news(newsroom_dir, 'train')
    dev = read_news(newsroom_dir, 'dev')
    test = read_news(newsroom_dir, 'test')

    # Doing some sort of pre-processing to remove unnecessary symbols, and then tokenizing...
    train = preprocess_tokenize(train)
    dev = preprocess_tokenize(dev)
    test = preprocess_tokenize(test)

    # Dumping news to jsonl files...
    write_jsonl(train, 'train')
    write_jsonl(dev, 'dev')
    write_jsonl(test, 'test')


    with open(newsroom_dir, mode='r') as nr:
        for line in nr:
            news = json.loads(line)
