import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class DebertaRationalityScorer(nn.Module):
    """DeBERTa合理性评分模型"""
    
    def __init__(self, model_name="./deberta-v3-large"):
        
        super().__init__()
        
        # 加载配置并设置为回归任务
        config = AutoConfig.from_pretrained(model_name)
        # config.num_labels = 1
        # config.problem_type = "regression"
        
        self.config = config
        
        # 加载预训练模型
        # self.deberta = AutoModel.from_pretrained(model_name, config=config)
        
        self.deberta = AutoModel.from_pretrained(model_name)
        config = self.deberta.config
        
        
        # super().__init__.from_pretrained(model_name, config=config)
        
        # 构建回归器替换原有分类器
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(), 
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # self.classifier = self.regressor
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, num_items_in_batch=None, return_dict=None, **kwargs):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            **kwargs
        )
        
        pooled_output = outputs[0][:, 0, :]  # 使用[CLS] token
        logits = self.classifier(pooled_output)
        
        # 将输出从[0,1]映射到[1,5]
        score = logits * 4 + 1
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(score.squeeze(), labels)
        
        if not return_dict:
            output = (score,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )