import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
# 确保 ORDataset, tokenizer, model 已经可用

# ----------------------------------------------------------------------
# A. 辅助函数：自定义 Collate Fn (处理字符串 ID)
# ----------------------------------------------------------------------
def inference_collate_fn(batch):
    """用于推理的 Collate 函数：处理张量和非张量数据 (如 ID)"""
    
    # 提取 ID (不能转换为张量)
    ids = [item.pop("id") for item in batch]
    
    # 将剩余的张量数据使用 PyTorch 默认方式堆叠
    # 注意：我们必须确保 'avg_raw' 和 'stdev_raw' 等非模型输入字段也被正确堆叠，
    # 否则 DataLoader 会报错，但我们在推理中可以忽略它们。
    try:
        # 使用 PyTorch 默认 collate 处理张量
        tensor_batch = torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        # 如果默认 collate 失败，手动处理
        # 简单起见，这里假设只有 input_ids 和 attention_mask 是模型所需
        tensor_batch = {k: torch.stack([item[k] for item in batch]) for k in batch[0] if k != 'id'}

    # 重新加入 ID
    tensor_batch["id"] = ids
    return tensor_batch


# ----------------------------------------------------------------------
# 路径和配置
# ----------------------------------------------------------------------
TEST_JSON_PATH = "../data/dev.json" 
INFERENCE_BATCH_SIZE = 32
OUTPUT_DIR = "./output_ordinal_regression"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSONL_PATH = os.path.join(OUTPUT_DIR, "dev_predictions_mean.jsonl")

# 确定设备
# ⚠️ 确保 model 和 tokenizer 在此处被加载，否则 device 引用会失败
# 假设 model 已经被加载
device = model.device 
model.eval() 

print(f"Model is on: {device}")
print(f"Using Test/Inference file: {TEST_JSON_PATH}")

# ----------------------------------------------------------------------
# 1. 实例化推理数据集和 DataLoader
# ----------------------------------------------------------------------
dev_dataset = ORDataset(
    json_path=TEST_JSON_PATH, 
    tokenizer=tokenizer, 
    mode='predict',
)
dev_dataloader = DataLoader(
    dev_dataset,
    sampler=SequentialSampler(dev_dataset),
    batch_size=INFERENCE_BATCH_SIZE,
    collate_fn=inference_collate_fn # 使用自定义 collate_fn 修复 ID 问题
)

# ----------------------------------------------------------------------
# 2. 运行推理循环
# ----------------------------------------------------------------------
all_results = []
# 类别值 (1-5) 权重
score_values = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device=device).unsqueeze(0)

print("\n***** 开始批量手动推理 (期望值预测) *****")

with torch.no_grad():
    for batch in tqdm(dev_dataloader, desc="Inferencing"):
        
        # 提取输入和 ID
        ids = batch.pop("id") 
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # 运行模型
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 提取预测值：likelihoods (N, 5)
        # 模型的 forward 返回的是 (y_star, likelihoods) 
        if isinstance(outputs, tuple):
            likelihoods = outputs[-1] # 始终是最后一个元素
        else:
            raise ValueError("模型输出格式错误，请检查模型 forward 方法的返回类型。")

        # 1. 计算期望值 (Mean Prediction)
        # E[Y] = sum_{k=1}^5 k * P(Y=k)
        # likelihoods (N, 5) * score_values (1, 5) -> (N, 5)
        mean_predictions = (likelihoods * score_values).sum(dim=1) # (N)
        
        # 2. 四舍五入到最近的整数 (1-5)
        # .round() 四舍五入到最接近的整数
        final_scores_float = mean_predictions.cpu().numpy()
        final_scores = np.round(final_scores_float).astype(int)
        
        # 确保分数在 1 到 5 之间
        final_scores = np.clip(final_scores, 1, 5) 
        
        # 收集结果
        for json_key, score in zip(ids, final_scores.tolist()):
            all_results.append({
                "id": json_key, 
                "prediction": int(score)
            })

print("\n手动推理完成。")

# ----------------------------------------------------------------------
# 3. 保存为 JSON Lines (.jsonl)
# ----------------------------------------------------------------------
with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f:
    for result in all_results:
        f.write(json.dumps(result) + '\n')

print(f"✅ 所有预测结果已保存到 {OUTPUT_JSONL_PATH}")