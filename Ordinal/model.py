import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
import torch

import torch
import torch.nn as nn
from torch.nn.init import uniform_
import torch.nn.functional as F
from transformers import DebertaV2Model, DebertaV2PreTrainedModel

# 损失函数
class OrdinalRegressionLoss(nn.Module):
    """
    基于累积 Logit 的序数回归损失函数（负对数似然）。
    接受有序的截止点和潜变量 Y*。
    """
    def __init__(self, num_class=5, train_cutpoints=False, scale=20.0):
        super().__init__()
        self.num_classes = num_class
        
        # 分割点 = 类别 - 1
        num_cutpoints = self.num_classes - 1
        
        # 初始化切点：均匀分布在[-scale/2, scale/2]区间
        self.cutpoints = torch.arange(num_cutpoints).float()*scale/(num_class-2) - scale / 2
        
        # 转换成可训练张量
        self.cutpoints = nn.Parameter(self.cutpoints)
        
        # 切点训练开关
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)

    def forward(self, pred, labels):
        
        sigmoids = torch.sigmoid(self.cutpoints - pred) 
        
        # P(Y=k) = P(Y<=k) - P(Y<=k-1)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        
        link_mat = torch.cat((
                sigmoids[:, [0]],    # P(类别=0) = σ(c0-pred)
                link_mat,    # P(类别=1,2,3)
                (1 - sigmoids[:, [-1]])    # P(类别=4) = 1 - σ(c3-pred)
            ),
            dim=1
        )
        
        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)  # 防止log(0)
        
        neg_log_likelihood = torch.log(likelihoods)  # 对数似然
        
        if labels is None:
            loss = 0
        else:
            loss = -torch.gather(neg_log_likelihood, 1, labels).mean()
            
        return loss, likelihoods



# ----------------------------------------------------------------------
# B. DeBERTaV2 序数回归模型
# ----------------------------------------------------------------------

class DebertaV2ForOrdinalRegression(DebertaV2PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.deberta = DebertaV2Model(config)
        
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, 1)
        )
        
        # 有序回归损失函数
        self.ordinal_loss = OrdinalRegressionLoss(
            num_class=5,    # 1-5分，共5个类别
            train_cutpoints=True,  # 允许学习切点
            scale=10.0              # 中等范围，可根据数据调整
        )
        
        self.init_weights()
        
    def get_cutpoints(self):
        """获取当前的切点值"""
        return self.ordinal_loss.cutpoints.data.cpu().numpy()

    # ----------------------------------------------------------------------

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        
        # 编码
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        
        X = outputs[0][:, 0, :] 
        
        # 1. 预测潜变量 Y*
        y_star = self.regressor(X) # 形状: [B, 1]
        
        
        # 2. 损失计算
        if labels is not None:
            # 确保labels形状正确 [batch_size, 1]
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            
            loss, probs = self.ordinal_loss(y_star, labels)
        else:
            loss = None
            _, probs = self.ordinal_loss(y_star, None)
             
        return {
            "loss": loss,
            "logits": probs,           # 类别概率 [batch_size, 5]
            "y_star": y_star           # 潜变量预测 [batch_size, 1]
        }
    
    def predict_score(self, input_ids, attention_mask=None):
        """预测1-5分"""
        with torch.no_grad():
            outputs = self(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs["logits"]  # [batch_size, 5]
            pred_class = torch.argmax(probs, dim=-1)  # 0-4
            return pred_class + 1  # 转换为1-5分