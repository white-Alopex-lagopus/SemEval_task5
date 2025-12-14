import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Model, DebertaV2PreTrainedModel


class DebertaV2ForOrdinalRegression(DebertaV2PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = 5
        self.deberta = DebertaV2Model(config)
        
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, self.num_classes - 1)    # 4
        )


    # ----------------------------------------------------------------------

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        
        # 编码
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        
        X = outputs[0][:, 0, :] 
        
        # 1. 预测潜变量 Y*
        logits = self.regressor(X) # 形状: [batch_size, 4]
        
        # 2. 损失计算
        if labels is not None:
            labels_float = labels.float().unsqueeze(1)
            
            
            
            # 构造阈值张量: [1, 2, 3, 4]
            thresholds = torch.arange(1, self.num_classes, dtype=torch.float, device=labels.device).unsqueeze(0)
            
            # T 是二元目标标签，形状 [B, 4]
            target_labels = (labels_float > thresholds).float()
            
            # 3. 计算 BCE 损失
            loss_bce = F.binary_cross_entropy_with_logits(logits, target_labels)
            std_devs = logits.std(dim=0)
            loss_stddev = -std_devs.mean()
            lambda_reg = 0.01
            loss = loss_bce + lambda_reg * loss_stddev
            # loss = F.binary_cross_entropy_with_logits(logits, target_labels) # 使用功能强大的 F.binary_cross_entropy_with_logits
            
        else:
            loss = None
             
        # 这里的 logits 是 K-1 个 Logits，而不是最终分数
        return (loss, logits) if loss is not None else logits
    
    
    
    def predict_score(self, input_ids, attention_mask):
        # 获取 logits (形状: [B, 4])
        # 确保 predict_score 是类方法
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)[1] if isinstance(self.forward(input_ids, attention_mask), tuple) else self.forward(input_ids, attention_mask)
        
        # 1. 转换为概率 P(Y > j)
        probs = torch.sigmoid(logits) # 形状: [B, 4]

        # 2. 计算预测分数
        # 分数 = 1 + 所有概率 > 0.5 的数量之和
        predicted_score = torch.sum(probs > 0.5, dim=1) + 1 
        
        return predicted_score.long()