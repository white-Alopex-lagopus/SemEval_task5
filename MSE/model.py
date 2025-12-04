import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
import torch

# 输出维度为 1
NUM_OUTPUTS = 1 

class DebertaV2ForWSDScoring(DebertaV2PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        
        # 回归头，输出维度为 1
        self.regressor = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob), 
            nn.Linear(config.hidden_size, NUM_OUTPUTS) 
        )
        self.post_init() 

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # ... (DeBERTaV2 主体运行部分不变) ...
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        cls_output = outputs[0][:, 0, :]
        predictions = self.regressor(cls_output) 

        loss = None
        if labels is not None:
            # 使用标准的 nn.MSELoss
            loss_fct = nn.MSELoss() 
            
            # labels 是 float32 类型的 average (T)
            # 确保 labels 的形状与 predictions 匹配 (batch_size, 1)
            target = labels.float().view(-1, NUM_OUTPUTS)
            loss = loss_fct(predictions.view(-1, NUM_OUTPUTS), target) # predictions.view(-1, 1)

        # 返回结果 (loss, predictions) 或 predictions
        return (loss, predictions) if loss is not None else predictions