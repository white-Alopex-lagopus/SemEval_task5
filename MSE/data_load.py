import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from torch.utils.data import Dataset
from transformers import DebertaV2Tokenizer

class MSEDataset(Dataset):
    """用于回归评分任务的WSD数据集"""
    
    def __init__(self, json_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 1. 读取JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 构造样本：每个原始条目（上下文/义项对）作为一个样本
        for key, item in data.items():
            
            # --- 文本信息 ---
            homonym = item["homonym"]
            definition = item["judged_meaning"]
            example = item["example_sentence"]
            # 完整上下文
            context = f"{item['precontext']} {item['sentence']} {item['ending']}"
            
            # --- 标签信息 ---
            # 直接使用平均值 (avg) 作为回归目标 T
            target_avg = item["average"] 
            # 使用标准差 (stdev) 作为损失函数中的容忍度 sigma
            target_stdev = item["stdev"]
            
            # 确保 avg 和 stdev 是有效的浮点数
            if target_avg is None or target_stdev is None:
                # 实际应用中可能需要更复杂的缺失值处理
                continue 
                
            self.samples.append({
                "json_key": key,
                "homonym": homonym,
                "definition": definition,
                "example": example,
                "context": context,
                "target_avg": target_avg,   # 平均分 (T)
                "target_stdev": target_stdev, # 标准差 (sigma)
                "sample_id": item['sample_id'] # 原始ID
            })
            
        print(f"创建了 {len(self.samples)} 个回归训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 构造输入文本
        text_parts = (
            f"homonym：{sample['homonym']}",
            f"Definition:{sample['definition']}",
            f"Example:{sample['example']}",
            f"Context:{sample['context']}"
        )
        # 使用tokenizer的sep_token连接各个部分
        text = self.tokenizer.sep_token.join(text_parts)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移除batch维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 重点修改：添加两个回归标签
        
        # 1. average 作为 labels (T)
        encoding["labels"] = torch.tensor(sample["target_avg"], dtype=torch.float32)
        
        # 2. stdev 作为 stdevs (sigma)，用于自定义损失函数
        # 自定义时加重惩罚区间外的，均方误差（MSELoss时不需要）
        # encoding["stdevs"] = torch.tensor(sample["target_stdev"], dtype=torch.float32)
        
        encoding["id"] = sample["json_key"]
        
        if "token_type_ids" in encoding:
            del encoding["token_type_ids"]
            
        return encoding

# 使用示例
if __name__ == "__main__":
    # 假设您的tokenizer路径正确
    # tokenizer = DebertaV2Tokenizer.from_pretrained("../deberta-v3-large") 
    
    # 临时使用一个通用的DebertaV2/V3 tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained("I:\deberta-v3-large")
    
    # 假设您的数据路径正确
    # dataset = SimpleWSDDataset("../data/train.json", tokenizer) 
    
    # 为了让代码可以运行，我们先创建一个虚拟的json文件
    dummy_data = {
        "8": {
            "homonym": "dribbling",
            "judged_meaning": "propel,",
            "precontext": "Tommy was excited...",
            "sentence": "He had been dribbling constantly.",
            "ending": "A small trail of wetness...",
            "choices": [4, 1, 1, 2, 3],
            "average": 2.2, # <--- Target T
            "stdev": 1.3038404810405297, # <--- Target Sigma
            "nonsensical": [False, False, True, False, False],
            "sample_id": "147",
            "example_sentence": "He likes to dribble the basketball."
        }
        # 您的真实数据中应该有更多条目
    }
    with open("dummy_train.json", 'w') as f:
        json.dump(dummy_data, f)
        
    dataset = MSEDataset("dummy_train.json", tokenizer)
    
    # 查看第一个样本
    sample = dataset[0]
    print(f"\n--- 样本信息 ---")
    print(f"输入 shape: {sample['input_ids'].shape}")
    print(f"标签 (Avg): {sample['labels'].item():.4f}")
    print(f"Stdev (Sigma): {sample['stdevs'].item():.4f}")
    
    # 解码看看
    text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print("\n输入文本:")
    # print(text)
    print(text[:300] + "...")