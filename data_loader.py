import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class HomonymDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.examples = self._preprocess_data()
    
    def _preprocess_data(self):
        """预处理原始数据"""
        examples = []
        
        for item_id, item in self.raw_data.items():
            # 构建输入文本
            input_text = f"{item['precontext']} {item['sentence']} {item['ending']}".strip()
            
            # 使用平均分作为回归目标
            target_score = float(item["average"])
            
            example = {
                "input_text": input_text,
                "score": target_score,
                "item_id": int(item_id),  # 转换为整数以匹配评估脚本
                "raw_data": item  # 保存原始数据用于调试
            }
            examples.append(example)
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 编码文本
        encoding = self.tokenizer(
            example["input_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(example["score"], dtype=torch.float),
            "item_id": example["item_id"]
        }

def collate_fn(batch):
    """自定义批处理函数"""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        # "item_ids": [item["item_id"] for item in batch]
    }