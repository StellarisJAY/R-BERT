import os
from transformers import AutoTokenizer, BertConfig
from torch.utils.data import DataLoader
import torch
from dataset import load_relations, RelationDataset
from model import RBERT
from tqdm import tqdm
import yaml

def train(model: RBERT, train_data: DataLoader, valid_data: DataLoader, device: torch.device, loss, optim):
    train_losses = []
    train_accs = []
    for data in tqdm(train_data):
        optim.zero_grad()
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        e1_mask, e2_mask = data['e1_mask'].to(device), data['e2_mask'].to(device)
        label = data['label'].to(device)
        y_pred = model(input_ids, attention_mask, e1_mask, e2_mask)
        l = loss(y_pred, label)
        l.backward()
        optim.step()
        with torch.no_grad(): 
            train_losses.append(l)
            train_accs.append(accuracy(y_pred, label))

    with torch.no_grad():
        print(f'avg train loss = {torch.tensor(train_losses).mean()}, acc={torch.tensor(train_accs).mean()*100:.2f}%')
        valid_loss, valid_acc = valid(model, valid_data, device, loss)
        print(f'avg valid loss = {valid_loss}, acc={valid_acc*100:.2f}%')

def valid(model: RBERT, valid_data: DataLoader, device: torch.device, loss):
    valid_losses = []
    valid_accs = []
    for data in tqdm(valid_data):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        e1_mask, e2_mask = data['e1_mask'].to(device), data['e2_mask'].to(device)
        label = data['label'].to(device)
        y_pred = model(input_ids, attention_mask, e1_mask, e2_mask)
        l = loss(y_pred, label)
        valid_losses.append(l)
        valid_accs.append(accuracy(y_pred, label))
    return (torch.tensor(valid_losses).mean(), torch.tensor(valid_accs).mean())


def accuracy(y_pred: torch.Tensor, y: torch.Tensor):
    y_pred = y_pred.argmax(dim=-1)
    return (y_pred == y).float().mean()

def test(model: RBERT, test_data: DataLoader, device: torch.device, loss):
    test_losses = []
    test_accs = []
    for data in tqdm(test_data):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        e1_mask, e2_mask = data['e1_mask'].to(device), data['e2_mask'].to(device)
        label = data['label'].to(device)
        y_pred = model(input_ids, attention_mask, e1_mask, e2_mask)
        l = loss(y_pred, label)
        test_losses.append(l)
        test_accs.append(accuracy(y_pred, label))
    return (torch.tensor(test_losses).mean(), torch.tensor(test_accs).mean())

if __name__ == '__main__':
    with open('./config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), yaml.FullLoader)
    
    MODEL_PATH = config['model_dir']
    train_file = os.path.join(config['data_dir'], 'train.csv')
    valid_file = os.path.join(config['data_dir'], 'valid.csv')
    test_file = os.path.join(config['data_dir'], 'test.csv')
    relation_file = os.path.join(config['data_dir'], 'relation.csv')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 加载关系定义
    relations = load_relations(relation_file)
    # 创建模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_config = BertConfig.from_pretrained(MODEL_PATH, num_hidden_layers=config['num_hidden_layers'])
    model = RBERT(MODEL_PATH, model_config, device=device, num_labels=len(relations), dropout=config['dropout'])
    print(model)
    # 加载数据集
    train_data = RelationDataset(train_file, tokenizer, relations, config['max_seq_len'])
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    valid_data = RelationDataset(valid_file, tokenizer, relations, config['max_seq_len'])
    valid_dataloader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=False)

    test_data = RelationDataset(test_file, tokenizer, relations, config['max_seq_len'])
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
    # 训练模型
    model.train()
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=1e-5)
    for epoch in range(config['epochs']):
        train(model, train_dataloader, valid_dataloader, device, loss, optim)
    # 测试模型
    model.eval()
    with torch.no_grad():
        test_loss, test_acc = test(model, test_dataloader, device, loss)
        print(f'test loss = {test_loss}, acc={test_acc*100:.2f}%')

    # 保存模型参数
    model.save(config['save_model_dir'])
    tokenizer.save_pretrained(config['save_model_dir'])
    