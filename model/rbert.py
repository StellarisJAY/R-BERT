import torch
from torch import nn
from transformers import BertModel, BertConfig

class FC(nn.Module):
    def __init__(self, input_size, output_size, activate=False, dropout=0.0, device: torch.device=None):
        super(FC, self).__init__()
        self.dense = nn.Linear(input_size, output_size, device=device)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.activate = activate
    
    def forward(self, x):
        x = self.dropout(x)
        if self.activate:
            x = self.tanh(x)
        return self.dense(x)


class RBERT(torch.nn.Module):
    def __init__(self, model_path: str, config: BertConfig, device: torch.device, num_labels=10, dropout=0.0):
        super(RBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_path, config=config)
        self.bert.to(device)
        self.dropout = nn.Dropout(dropout)
        
        self.fc_cls = FC(config.hidden_size, config.hidden_size, dropout=dropout, device=device, activate=True)
        self.fc_entity = FC(config.hidden_size, config.hidden_size, dropout=dropout, device=device, activate=True)

        self.classifier = FC(
            config.hidden_size * 3,
            num_labels,
            activate=False,
            dropout=dropout,
            device=device
        )
    
    def entity_average(self, hidden_state: torch.Tensor, entity_mask: torch.Tensor):
        '''
        param hidden_state: bert last_hidden_state, shape=(B,n,d)
        param entity_mask:  [0,0,1,1,0,0,0], shape=(B,n)
        return: state, shape=(B,n,d)
        '''
        # (B, 1, n)
        e_mask = entity_mask.unsqueeze(1)
        # (B, 1)
        n = entity_mask.sum(dim=1).unsqueeze(1)
        # (B, 1, d) -> (B, d)
        sum = torch.bmm(e_mask, hidden_state).squeeze(1)
        return sum / n

    def forward(self, input_ids, input_mask, e1_mask, e2_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=input_mask)
        last_hidden_state = bert_output.last_hidden_state
        pooler_out = bert_output.pooler_output

        # (B,n,d)
        cls = self.fc_cls(pooler_out)
        e1_avg = self.entity_average(last_hidden_state, e1_mask)
        # (B, d)
        e1 = self.fc_entity(e1_avg)
        e2_avg = self.entity_average(last_hidden_state, e2_mask)
        # (B, d)
        e2 = self.fc_entity(e2_avg)
        
        concat = torch.cat([cls.unsqueeze(1), e1.unsqueeze(1), e2.unsqueeze(1)], dim=1).reshape(cls.shape[0], -1)
        return self.classifier(concat)
