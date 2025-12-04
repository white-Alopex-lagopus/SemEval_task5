import json
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm
import os
from model import DebertaV2ForWSDScoring
from transformers import DebertaV2Tokenizer
from data_load import MSEDataset
# from data_load import SimpleWSDDataset # 确保您的 Dataset 类可以访问

# ----------------------------------------------------------------------
# 路径和配置
# ----------------------------------------------------------------------
DEV_JSON_PATH = "/kaggle/input/semeval/data/dev.json" # 开发集数据路径
INFERENCE_BATCH_SIZE = 32 
OUTPUT_JSONL_PATH = "dev_predictions.jsonl"

# 确定设备 (假设 model 已经移动到正确的设备)
device = model.device 
model.eval() # 切换到评估模式

print(f"Model is on: {device}")
print(f"Using JSON file: {DEV_JSON_PATH}")

# ----------------------------------------------------------------------
# 1. 实例化推理数据集和 DataLoader
# ----------------------------------------------------------------------
# 实例化 SimpleWSDDataset，传入 is_training=False 禁用标签加载
dev_dataset = SimpleWSDDataset(
    json_path=DEV_JSON_PATH, 
    tokenizer=tokenizer, 
    is_training=False
) 
dev_dataloader = DataLoader(
    dev_dataset,
    sampler=SequentialSampler(dev_dataset),
    batch_size=INFERENCE_BATCH_SIZE
)

# ----------------------------------------------------------------------
# 2. 运行推理循环
# ----------------------------------------------------------------------
all_results = []
print("\n***** 开始批量推理 *****")

with torch.no_grad():
    for batch in tqdm(dev_dataloader, desc="Inferencing"):
        
        # 提取 ID 列表和模型输入
        # 假设 SimpleWSDDataset 返回的 batch 字典中包含 'id' 键
        ids = batch["id"] 
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # 运行模型
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 提取预测值 (logits/scores)
        predictions = outputs[0] if isinstance(outputs, tuple) else outputs
        predicted_scores = predictions.squeeze().cpu().tolist()
        
        # 收集结果
        for json_key, score in zip(ids, predicted_scores): 
            # 最终评分四舍五入并限制在 [1.0, 5.0]
            final_score = round(max(1.0, min(5.0, score)))
            
            all_results.append({
                "id": json_key, 
                "prediction": final_score
            })

print("\n推理完成。")

# ----------------------------------------------------------------------
# 3. 保存为 JSON Lines (.jsonl)
# ----------------------------------------------------------------------

print(f"开始保存 {len(all_results)} 条结果到 {OUTPUT_JSONL_PATH}...")

with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f:
    for result in all_results:
        # 将字典序列化为 JSON 字符串，并写入一行
        f.write(json.dumps(result) + '\n')

print(f"所有预测结果已保存到 {OUTPUT_JSONL_PATH}")

# 打印全部结果
print("\n--- 全部预测结果 (JSON Lines 格式) ---")
for result in all_results:
    print(f"{result}")