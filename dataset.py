from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import csv

def load_relations(path: str):
    with open(path, 'r', encoding='utf-8') as f:
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

def preprocess_data(data_rows: list[list[str]], tokenizer: AutoTokenizer, relations: dict, max_length=256):
    data = []
    for row in data_rows:
        [sentence, rel, e1, e1_off, e2, e2_off] = row
        e1_off, e2_off = int(e1_off), int(e2_off)
        # inputs = tokenizer(sentence, return_offsets_mapping=True, max_length=max_length, padding='max_length', truncation='longest_first')
        # e1_mask = [0] * len(inputs['input_ids'])
        # e2_mask = [0] * len(inputs['input_ids'])
        # idx = 0
        # for m in inputs['offset_mapping']:
        #     (start, end) = m
        #     if start == 0 and end == 0:
        #         idx += 1
        #         continue
        #     if (start >= e1_off and start < e1_off + len(e1)):
        #         e1_mask[idx] = 1
        #     if (start >= e2_off and start < e2_off + len(e2)):
        #         e2_mask[idx] = 1
        #     idx += 1
        # data.append({
        #     'input_ids': torch.tensor(inputs['input_ids']),
        #     'attention_mask': torch.tensor(inputs['attention_mask']),
        #     'e1_mask': torch.tensor(e1_mask).float(),
        #     'e2_mask': torch.tensor(e2_mask).float(),
        #     'label': torch.tensor(relations.get(rel)['index']),
        # })
        data.append(preprocess_row(sentence, e1, e2, e1_off, e2_off, rel, tokenizer, relations, max_length))
    return data

def preprocess_row(sentence: str, e1: str, e2: str, e1_off: int, e2_off: int, rel: str, tokenizer: AutoTokenizer, relations: dict, max_length=256):
    inputs = tokenizer(sentence, return_offsets_mapping=True, max_length=max_length, padding='max_length', truncation='longest_first')
    e1_mask = [0] * len(inputs['input_ids'])
    e2_mask = [0] * len(inputs['input_ids'])
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
    return {
        'input_ids': torch.tensor(inputs['input_ids']),
        'attention_mask': torch.tensor(inputs['attention_mask']),
        'e1_mask': torch.tensor(e1_mask).float(),
        'e2_mask': torch.tensor(e2_mask).float(),
        'label': torch.tensor(relations.get(rel)['index']),
    }

class RelationDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, relations, max_length=256):
        super(RelationDataset, self).__init__()
        with open(data_file, 'r', encoding='utf-8') as f:
            data = csv.reader(f.readlines())
            data = [row for row in data][1:]
        self.data = preprocess_data(data, tokenizer, relations,  max_length)

    def __getitem__(self, index):
        data = self.data[index]
        return data
    
    def __len__(self):
        return len(self.data)
