import os
from transformers import AutoTokenizer, BertConfig
import torch
from dataset import load_relations, preprocess_row
from model import RBERT

DATA_DIR = './data'
MODEL_PATH = "./trained_model"

def predict(model: RBERT, text: str, e1: str, e2: str, relations: dict, tokenizer: AutoTokenizer, device: torch.device):
    model.eval()
    e1_off = text.index(e1)
    e2_off = text.index(e2)
    inputs = preprocess_row(text, e1, e2, e1_off, e2_off, 'None', tokenizer, relations, 256)
    input_ids = inputs['input_ids'].unsqueeze(0).to(device)
    attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
    e1_mask, e2_mask = inputs['e1_mask'].unsqueeze(0).to(device), inputs['e2_mask'].unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred = model(input_ids, attention_mask, e1_mask, e2_mask)
        y_pred = torch.argmax(y_pred, dim=1)
        return y_pred.item()

if __name__ == '__main__':
    relation_file = os.path.join(DATA_DIR,'relation.csv')
    relations = load_relations(relation_file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = BertConfig.from_pretrained(MODEL_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RBERT(MODEL_PATH, config, device, len(relations), 0.0)

    result = predict(model, '明朝末年抗清英雄黄得功，本姓王，安徽合肥人后改姓黄', '黄得功', '安徽合肥', relations, tokenizer, device)
    print(result)