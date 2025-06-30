import os
from transformers import AutoTokenizer, BertConfig
import torch
from dataset import load_relations, preprocess_row
from model import RBERT
import yaml

def predict(model: RBERT, text: str, e1: str, e2: str, relations: dict, tokenizer: AutoTokenizer, device: torch.device):
    model.eval()
    e1_off = text.index(e1)
    e2_off = text.index(e2)
    inputs = preprocess_row(text, e1, e2, e1_off, e2_off, 'None', tokenizer, relations, 256)
    input_ids = inputs['input_ids'].unsqueeze(0).to(device) # (1, n)
    attention_mask = inputs['attention_mask'].unsqueeze(0).to(device) # (1, n)
    e1_mask, e2_mask = inputs['e1_mask'].unsqueeze(0).to(device), inputs['e2_mask'].unsqueeze(0).to(device) # (1, n)
    with torch.no_grad():
        y_pred = model(input_ids, attention_mask, e1_mask, e2_mask) # (1, num_labels)
        y_pred = torch.nn.functional.softmax(y_pred, dim=-1)
        confidence = y_pred.max(dim=-1).values
        y_pred = torch.argmax(y_pred, dim=-1)
        return y_pred.item(), confidence.item()

if __name__ == '__main__':
    with open('./config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), yaml.FullLoader)
    
    MODEL_PATH = config['save_model_dir']
    relation_file = os.path.join(config['data_dir'],'relation.csv')
    relations = load_relations(relation_file)
    idx_to_rel = [{}] * len(relations)
    for k in relations:
        relation = relations[k]
        idx_to_rel[int(relation['index'])] = {
            'head_type': relation['head_type'],
            'tail_type': relation['tail_type'],
            'rel': k
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_config = BertConfig.from_pretrained(MODEL_PATH, num_hidden_layers=config['num_hidden_layers'])
    model = RBERT(MODEL_PATH, model_config, device, len(relations), 0.0)
    model.load(config['save_model_dir'], device)

    text = '张三是成都人，于2000年1月出生。目前张三在一个建筑工地工作。可以通过手机号13555556666联系到他。'
    e1 = '张三'
    e2s = ['2000年1月', '13555556666', '建筑工地']

    print(text)
    for e2 in e2s:
        label, confidence = predict(
            model=model,
            text=text,
            e1=e1,
            e2=e2,
            relations=relations,
            tokenizer=tokenizer,
            device=device
        )
        rel = idx_to_rel[label]['rel']
        print(f'关系：{e1}-{rel}->{e2}，置信度: {confidence}')