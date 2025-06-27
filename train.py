import os
import csv
from transformers import AutoTokenizer

DATA_DIR = './data'
MODEL_PATH = "D:\\models\\chinese-bert-wwm"

train_file = os.path.join(DATA_DIR, 'train.csv')
valid_file = os.path.join(DATA_DIR, 'valid.csv')
test_file = os.path.join(DATA_DIR, 'test.csv')

def load_relations():
    with open(os.path.join(DATA_DIR, 'relation.csv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f.readlines())
    
    rows = [row for row in reader][1:]
    relations = {}
    for row in rows:
        relations[row[2]] = {
            'head_type': row[0],
            'tail_type': row[1],
            'index': int(row[3])
        }
    return relations


with open(train_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f.readlines())


def preprocess_data(data_rows: list[list[str]], tokenizer: AutoTokenizer, relations: dict):
    data = []
    for row in data_rows:
        [sentence, rel, e1, e1_off, e2, e2_off] = row
        inputs = tokenizer(sentence, return_offsets_mapping=True)
        e1_mask = [0] * len(inputs['input_ids'])
        e2_mask = [0] * len(inputs['input_ids'])
        e1_off, e2_off = int(e1_off), int(e2_off)
        idx = 0
        for m in inputs['offset_mapping']:
            (start, end) = m
            if start == 0 and end == 0:
                idx += 1
                continue
            if (start >= e1_off and start < e1_off + len(e1)):
                e1_mask[idx] = 1
            if (start >= e2_off and start < e2_off + len(e2)):
                e2_mask[idx] = 1
            idx += 1
        data.append({
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'e1_mask': e1_mask,
            'e2_mask': e2_mask,
            'label': relations.get(rel)['index'],
        })
    return data


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
relations = load_relations()
preprocess_data([row for row in reader][1:], tokenizer, relations)