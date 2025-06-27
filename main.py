from model import RBERT
from transformers import BertConfig, BertTokenizer
import torch

MODEL_PATH = "D:\\models\\chinese-bert-wwm"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = BertConfig.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = RBERT(
    model_path=MODEL_PATH,
    config=config,
    device=device,
    num_labels=5,
    dropout=0.1
)

x = torch.ones((2, 10)).long()
input_mask = torch.ones((2, 10)).long()
e1_mask = torch.ones((2, 10))
e2_mask = torch.ones((2, 10))

y_pred = model(x, input_mask, e1_mask, e2_mask)

print(y_pred.shape)


