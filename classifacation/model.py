import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel

# 类别数量是 5 （对应 1, 2, 3, 4, 5 分）
NUM_LABELS = 5 

class DebertaV2ForWSDScoring(DebertaV2PreTrainedModel):
    """
    继承 DeBERTaV2Model，添加一个输出5个类别的分类头，
    用于 1-5 分的合理性评分任务。
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 1. DeBERTaV2 主体 (用于特征提取)
        self.deberta = DebertaV2Model(config)
        
        # 2. 分类头 (Classification Head)
        self.classifier = nn.Sequential(
            # dropout rate从config中获取
            nn.Dropout(config.hidden_dropout_prob), 
            # 将 DeBERTaV2 的隐藏状态维度映射到 5 个类别
            nn.Linear(config.hidden_size, NUM_LABELS) 
        )
        
        # 初始化权重
        self.post_init() 

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        
        # 运行 DeBERTaV2 主体
        # DeBERTaV2 模型的 forward 默认不使用 token_type_ids，如果数据集中有，会被 **kwargs 吸收
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # 提取 [CLS] token 的隐藏状态（第一个 token 的输出）
        # (batch_size, hidden_size)
        cls_output = outputs[0][:, 0, :]
        
        # 运行分类头
        # (batch_size, NUM_LABELS)
        logits = self.classifier(cls_output) 

        loss = None
        if labels is not None:
            # 交叉熵损失函数
            # logits: (batch_size, 5)
            # labels: (batch_size,) 且值范围在 [0, 4]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))

        # 返回结果 (loss, logits) 或 logits
        return (loss, logits) if loss is not None else logits