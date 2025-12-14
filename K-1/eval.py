import numpy as np
import torch
from scipy.stats import spearmanr
from typing import Dict, Any, TYPE_CHECKING

# 仅用于类型提示，避免实际导入 EvalPrediction
if TYPE_CHECKING:
    from transformers import EvalPrediction 

def compute_metrics_final(p: 'EvalPrediction') -> Dict[str, float]:
    
    # --- 1. 提取预测分数 (Logits -> 0-4 整数) ---
    # p.predictions 是模型输出的 Logits (K-1 Logits, 形状 [N, 4])
    logits = torch.tensor(p.predictions)
    probs = torch.sigmoid(logits) 
    
    # 有序分类模型转为 0-4 标签
    predicted_labels_np = (torch.sum(probs > 0.5, dim=1)).numpy() # [N] (0-4 整数)

    # --- 2. 提取真实标签 (这是 ORDataset 在 eval 模式下返回的 0-4 整数) ---
    gold_labels_np = p.label_ids # [N] (0-4 整数)

    # --- A. Spearman Correlation (基于 0-4 整数标签) ---
    corr, _ = spearmanr(predicted_labels_np, gold_labels_np)
    
    # --- B. Accuracy within 1 (labels +/- 1 容忍度) ---
    abs_diff = np.abs(predicted_labels_np - gold_labels_np)
    acc_within_1 = np.sum(abs_diff <= 1) / len(predicted_labels_np)
    
    # --- C. 严格精度 (Accuracy == 0) ---
    strict_accuracy = np.sum(abs_diff == 0) / len(predicted_labels_np)

    # ❗ 必须移除对 p.metrics 的依赖，否则 Trainer 会报错
    # 参见之前的错误：AttributeError: 'EvalPrediction' object has no attribute 'metrics'
    
    return {
        "spearman_rho": float(corr),
        "acc_within_1": float(acc_within_1),
        "strict_accuracy": float(strict_accuracy),
        # "eval_loss" 将由 Trainer 自动添加到最终的评估结果中。
    }