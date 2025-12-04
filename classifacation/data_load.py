import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from torch.utils.data import Dataset
from transformers import DebertaV2Tokenizer

class SimpleWSDDataset(Dataset):
    """最简单的WSD数据集"""
    
    def __init__(self, json_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 1. 读取JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 展平：每个choice变成一个样本
        for key, item in data.items():
            homonym = item["homonym"]
            definition = item["judged_meaning"]
            example = item["example_sentence"]
            
            # 完整上下文
            context = f"{item['precontext']} {item['sentence']} {item['ending']}"
            
            # 为每个有效的choice创建一个样本
            for choice_idx, (score, nonsensical) in enumerate(zip(item["choices"], item["nonsensical"])):
                if not nonsensical:  # 只取有效的
                    self.samples.append({
                        "homonym": homonym,
                        "definition": definition,
                        "example": example,
                        "context": context,
                        "score": score,  # 1-5分
                        "sample_id": f"{item['sample_id']}_{choice_idx}"  # 唯一ID
                    })
        
        print(f"创建了 {len(self.samples)} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 最简单的输入格式：定义 + 例句 + 待判断文本
        # text = (
        #     f"homonym：{sample['homonym']} [SEP] "
        #     f"Definition:{sample['definition']} [SEP] "
        #     f"Example:{sample['example']} [SEP] "
        #     f"Context:{sample['context']}"
        # )
        
        text_parts = (
            f"homonym：{sample['homonym']}"
            f"Definition:{sample['definition']}"
            f"Example:{sample['example']}"
            f"Context:{sample['context']}"
        )
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
        
        
        
        # 添加labels
        # encoding["labels"] = torch.tensor(sample["score"], dtype=torch.float32)
        # 回归的时候再用这个
        
        
        score_index = sample["score"] - 1  # 转为0-4索引
        encoding["labels"] = torch.tensor(score_index, dtype=torch.long)
        
        if "token_type_ids" in encoding:
             del encoding["token_type_ids"]
             
        return encoding

# 使用示例
if __name__ == "__main__":
    # 初始化tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained("../deberta-v3-large")
    
    # 创建数据集
    dataset = SimpleWSDDataset("../data/train.json", tokenizer)
    
    # 查看第一个样本
    sample = dataset[0]
    print(f"输入shape: {sample['input_ids'].shape}")
    print(f"标签: {sample['labels'].item()}")
    
    # 解码看看
    text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print("\n输入文本:")
    print(text[:500])