import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from torch.utils.data import Dataset
from transformers import DebertaV2Tokenizer

class ORDataset(Dataset):
    """用于有序分类评分任务的WSD数据集"""
    
    def __init__(self, json_path, tokenizer, max_length=512, mode='train'):
        """
        Args:
            json_path (str): 数据文件路径.
            tokenizer (DebertaV2Tokenizer): 分词器.
            max_length (int): 最大序列长度.
            mode (str): 数据集模式 ('train', 'eval', 'predict').
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.samples = []
        
        # 1. 读取JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 构造样本
        for key, item in data.items():
            
            # --- 文本信息 ---
            homonym = item["homonym"]
            definition = item["judged_meaning"]
            example = item["example_sentence"]
            context = f"{item['precontext']} {item['sentence']} {item['ending']}"
            
            # --- 标签信息 ---
            target_avg = item.get("average", 0.0)
            target_stdev = item.get("stdev", 0.0)
            target_score = round(target_avg) 
            target_score = max(1, min(5, target_score)) # 确保在 1-5 范围内
            
            self.samples.append({
                "json_key": key,
                "homonym": homonym,
                "definition": definition,
                "example": example,
                "context": context,
                # 0 到 4 的整数标签
                "ordinal_label": target_score - 1, 
                "target_avg": target_avg,    
                "target_stdev": target_stdev,  
                "sample_id": item['sample_id']
            })
            
        print(f"✅ 创建了 {len(self.samples)} 个有序分类样本 (Mode: {self.mode})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 构造输入文本
        text_parts = (
            f"homonym：{sample['homonym']}",
            f"Definition:{sample['definition']}",
            f"Example:{sample['example']}",
            f"Context:{sample['context']}"
        )
        text = self.tokenizer.sep_token.join(text_parts)
        
        # 2. Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt" 
        )
        
        # 移除batch维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 移除 token_type_ids
        if "token_type_ids" in encoding:
            del encoding["token_type_ids"]
            
        # 3. 标签分配：根据模式返回不同的标签
        
        # 无论哪种模式，我们都需要 ID 
        # ID 作为字符串返回，需要 DataLoader/DataCollator 特殊处理
        encoding["id"] = sample["json_key"] 
        
        if self.mode in ['train', 'eval']:
            # 训练/验证模式：返回所有标签
            
            # 'labels' 是 Hugging Face Trainer 默认识别的主标签
            encoding["labels"] = torch.tensor(sample["ordinal_label"], dtype=torch.long)
            
            # 额外的回归和标准差，用于自定义加权损失（如果未使用，Trainer 会忽略）
            encoding["avg_raw"] = torch.tensor(sample["target_avg"], dtype=torch.float32)
            encoding["stdev_raw"] = torch.tensor(sample["target_stdev"], dtype=torch.float32)
            
        elif self.mode == 'predict':
            # 推理模式：只返回 ID 和 Tokenizer 输出
            # 注意：不返回 'labels' 或其他标签，避免 Trainer/DataLoader 尝试将其转换为 None/Tensor
            pass # 只需要文本输入和 ID
            
        return encoding