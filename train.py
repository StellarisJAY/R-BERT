import os
import csv
from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer

DATA_DIR = './data'
MODEL_PATH = "D:\\models\\chinese-bert-wwm"

train_file = os.path.join(DATA_DIR, 'train.csv')
valid_file = os.path.join(DATA_DIR, 'valid.csv')
test_file = os.path.join(DATA_DIR, 'test.csv')

with open(train_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f.readlines())

def preprocess_data(data_rows: list[list[str]], tokenizer: AutoTokenizer):
    data = []
    for row in data_rows[6:]:
        [sentence, rel, e1, e1_off, e2, e2_off] = row
        e1, e2 = row[3], row[5]
        inputs = tokenizer(sentence, return_offsets_mapping=True)
        e1_mask = [0] * len(inputs['input_ids'])
        e2_mask = [0] * len(inputs['input_ids'])
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'])
        print(tokens)
        e1_off, e2_off = int(e1_off), int(e2_off)
        e1_len, e2_len = len(e1), len(e2)
        for m in inputs['offset_mapping']:
            (start, end) = m
            if start == 0 and end == 0:
                continue
            if (start < e1_off or start > e1_off+len(e1)) and (start < e2_off or start > e2_off+len(e2)):
                continue
            print(sentence[start:end+1])
        break
    pass


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
preprocess_data([row for row in reader], tokenizer)