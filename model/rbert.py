import torch
from torch import nn
from transformers import BertModel, BertConfig
import os

class FC(nn.Module):
    def __init__(self, input_size, output_size, activate=False, dropout=0.0, device: torch.device=None):
        super(FC, self).__init__()
        self.dense = nn.Linear(input_size, output_size, device=device)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.activate = activate
    
    def forward(self, x):
        # 每个全连接层之前做dropout
        x = self.dropout(x)
        if self.activate:
            x = self.tanh(x)
        return self.dense(x)

# RBERT模型 参考：https://arxiv.org/abs/1905.08284 3.2 Model Architecture
class RBERT(torch.nn.Module):
    def __init__(self, model_path: str, config: BertConfig, device: torch.device, num_labels=10, dropout=0.0):
        super(RBERT, self).__init__()
        self.bert_config = config
        self.bert = BertModel.from_pretrained(model_path, config=config)
        self.bert.to(device)
        self.dropout = nn.Dropout(dropout)
        # [CLS] 位置的全连接层
        self.fc_cls = FC(config.hidden_size, config.hidden_size, dropout=dropout, device=device, activate=True)
        # 实体位置的全连接层
        self.fc_entity = FC(config.hidden_size, config.hidden_size, dropout=dropout, device=device, activate=True)
        # 最后的分类器全连接层
        self.classifier = FC(
            config.hidden_size * 3, # [CLS]和两个实体输出的拼接，最后的维度是 3xd
            num_labels,             # 输出的分类类型数量
            activate=False,
            dropout=dropout,
            device=device
        )

    # 计算实体的BERT输出平均值
    def entity_average(self, hidden_state: torch.Tensor, entity_mask: torch.Tensor):
        '''
        param hidden_state: bert last_hidden_state, shape=(B,n,d)
        param entity_mask:  [0,0,1,1,0,0,0], shape=(B,n)
        return: avergae_state, shape=(B,d)
        '''
        # (B, 1, n)
        e_mask = entity_mask.unsqueeze(1)
        # (B, 1)
        n = entity_mask.sum(dim=1).unsqueeze(1)
        # (B, 1, n) * (B, n, d) -> (B, 1, d) -> (B, d)
        # 矩阵乘法快速mask求和，mask行与state的列每一位相乘后相加，mask为1的保留为0的丢弃
        sum = torch.bmm(e_mask, hidden_state).squeeze(1)
        return sum / n

    def forward(self, input_ids, input_mask, e1_mask, e2_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=input_mask)
        last_hidden_state = bert_output.last_hidden_state # (B, n, d)
        pooler_out = bert_output.pooler_output #(B, d)
        
        # (B, d)
        cls = self.fc_cls(pooler_out)
        e1_avg = self.entity_average(last_hidden_state, e1_mask)
        # (B, d)
        e1 = self.fc_entity(e1_avg)
        e2_avg = self.entity_average(last_hidden_state, e2_mask)
        # (B, d)
        e2 = self.fc_entity(e2_avg)
        # (B, 3*d)
        concat = torch.cat([cls.unsqueeze(1), e1.unsqueeze(1), e2.unsqueeze(1)], dim=1).reshape(cls.shape[0], -1)
        return self.classifier(concat) # (B, num_labels)
    
    def load(self, path: str, device: torch.device):
        self.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin'), map_location=device))

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        model_file = os.path.join(path, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_file)
        self.bert_config.save_pretrained(path)
