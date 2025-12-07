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
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, cutpoints, label):
        # pred (Y*): 潜变量预测值，形状 [B, 1] 或 [B]
        # cutpoints (tau_k): 有序的截止点，形状 [K-1]
        
        # 确保 pred 形状为 [B, 1]，便于广播
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)

        # 1. 计算 Sigmoids (累积概率 P(Y <= k))
        # 形状 [B, K-1]
        sigmoids = torch.sigmoid(cutpoints - pred) 

        # 2. 计算 P(Y = k) 的概率
        # 类别概率 P(Y=k) = P(Y<=k) - P(Y<=k-1)
        
        # P(Y <= 0) = 0
        zeros = torch.zeros_like(sigmoids[:, :1]) 
        
        # P(Y <= K) = 1
        ones = torch.ones_like(sigmoids[:, :1])
        
        # 形状 [B, K+1]: [0, P(Y<=1), ..., P(Y<=K-1), 1]
        padded_cum_probs = torch.cat([zeros, sigmoids, ones], dim=1)
        
        # 类别概率 link_mat [B, K]
        # P(Y=k) = P(Y<=k) - P(Y<=k-1)
        likelihoods = padded_cum_probs[:, 1:] - padded_cum_probs[:, :-1]

        eps = 1e-15
        likelihoods = torch.clamp(likelihoods, eps, 1 - eps)

        # 3. 提取真实标签对应的负对数似然
        neg_log_likelihood = -torch.log(likelihoods)
        
        # 4. 提取真实标签的损失并求平均 (labels 是 0-indexed)
        if label is None:
            # 推理时只返回概率
            return torch.zeros(1, device=pred.device), likelihoods 
        else:
            # 训练时计算 NLL Loss
            loss = torch.gather(neg_log_likelihood, 1, label.unsqueeze(-1)).mean()
            return loss, likelihoods


# ----------------------------------------------------------------------
# B. DeBERTaV2 序数回归模型
# ----------------------------------------------------------------------

class DebertaV2ForOrdinalRegression(DebertaV2PreTrainedModel):
    NUM_CLASSES = 5 
    NUM_CUTPOINTS = NUM_CLASSES - 1 # 4

    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        
        # 1. 预测潜变量 Y* (维度为 1)
        self.linear_y_star = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1)
        )
        
        # 2. 截止点参数：
        #   a. base_cutpoint: 第一个截止点 (tau_1)
        #   b. cutpoint_deltas: 剩余截止点之间的间隔 (delta_k)
        
        # 初始化 tau_1 (例如均匀分布在 [-0.5, 0.5])
        self.base_cutpoint = nn.Parameter(uniform_(torch.empty(1), -0.5, 0.5))
        
        # 初始化 delta_k (必须是正数，通过 Softplus 保证)
        # 初始化为正数有助于快速收敛，我们只存储 K-2 个间隔 (tau_2 到 tau_4 的间隔)
        self.cutpoint_deltas = nn.Parameter(uniform_(torch.empty(self.NUM_CUTPOINTS - 1), 0.1, 0.5))
        
        self.custom_loss = OrdinalRegressionLoss(num_classes=self.NUM_CLASSES)
        self.post_init() 

    # ----------------------------------------------------------------------
    # 核心：计算有序截止点
    # ----------------------------------------------------------------------
    def get_ordered_cutpoints(self):
        """确保截止点有序(tau_1 < tau_2 < ...)"""
        
        # 1. 确保间隔为正 (Softplus/Exp)
        # Softplus(x) = log(1 + exp(x)), 保证输出 > 0
        deltas = F.softplus(self.cutpoint_deltas)
        
        # 2. 累加计算有序截止点
        # [tau_1] + [delta_2, delta_3, delta_4]
        # tau_k = tau_{k-1} + delta_k
        
        # 累积求和得到相对间隔
        cumulative_deltas = torch.cumsum(deltas, dim=0)
        
        # 形状 [1] + [K-2]
        # tau_1 = base_cutpoint
        # tau_2 = tau_1 + delta_2
        # tau_3 = tau_1 + delta_2 + delta_3 ...
        
        # 最终有序截止点 [tau_1, tau_2, tau_3, tau_4]
        ordered_cutpoints = torch.cat([
            self.base_cutpoint, 
            self.base_cutpoint + cumulative_deltas
        ], dim=0)
        
        return ordered_cutpoints

    # ----------------------------------------------------------------------

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        X = outputs[0][:, 0, :] 
        
        # 1. 预测潜变量 Y*
        y_star = self.linear_y_star(X) # 形状: [B, 1]

        # 2. 获取有序截止点
        ordered_cutpoints = self.get_ordered_cutpoints()

        loss = None
        # 3. 损失计算
        if labels is not None:
            loss, likelihoods = self.custom_loss(y_star, ordered_cutpoints, labels) 
        else:
             # 推理时只获取似然度
             _, likelihoods = self.custom_loss(y_star, ordered_cutpoints, labels=None)
             
        model_output = (y_star.squeeze(-1), likelihoods)
        
        if loss is not None:
            # 训练模式：(loss, output)
            return (loss,) + model_output
        else:
            # 推理模式：(output)
            return model_output



        # 训练时返回 Loss, 潜变量 Y*, 类别似然
        # return (loss, y_star.squeeze(-1), likelihoods) if loss is not None else (y_star.squeeze(-1), likelihoods)