import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class InsDataset(Dataset):
    
    def __init__(self, json_path, tokenizer, max_length=512, mode='train'):
        """
        Args:
            json_path (str): 数据文件路径.
            tokenizer (AutoTokenizer): 适用于生成模型的分词器
            max_length (int): 最大序列长度.
            mode (str): 数据集模式 ('train', 'eval', 'predict').
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.samples = []
        
        # 设置分词器的填充侧为左侧
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 1. 读取JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 构造样本
        for key, item in data.items():
            homonym = item["homonym"]
            definition = item["judged_meaning"]
            example = item["example_sentence"]
            context = f"{item['precontext']} {item['sentence']} {item['ending']}"
        
            target_avg = item.get("average", 0.0)
            target_score = max(1, min(5, (target_avg + 0.5).__floor__()))  # 传统四舍五入

            prompt = """你是一个语义消歧专家。你的任务是分析给定上下文中多义词的特定含义可能性。
请严格按照以下规则输出：
1. 仔细分析上下文、目标词和目标含义。
2. 输出必须是1到5之间的一个整数，不要包含任何其他文字、标点或解释。
3. 评分标准：1=非常不可能，2=不可能，3=中立，4=可能，5=非常可能。"""
            user_input = f"""请分析以下内容：

目标词：{homonym}
目标含义：{definition}
示例句子：{example}

上下文：
{context}

请根据上下文，判断目标词在此处携带上述目标含义的可能性。只输出一个1-5的整数。"""
        
            if mode != 'predict':
                messages = [
                   {"role": "system", "content": prompt},
                   {"role": "user", "content": user_input},
                   {"role": "assistant", "content": str(target_score)}
                ]
            else:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ]
        
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,      # 不进行分词
                add_generation_prompt=(mode == 'predict')
            )
        
            self.samples.append({
                "json_key": key,
                "full_text": full_text,
                "target_score": target_score,
                # "sample_id": item['sample_id']
            })
        
        print(f"创建了 {len(self.samples)} 个指令微调样本 (Mode: {self.mode})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 分词处理
        encoding = self.tokenizer(
            sample["full_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移除batch维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 对于生成模型，输入文本本身就是标签
        # 我们将input_ids复制给labels，模型将学习预测下一个token
        encoding["labels"] = encoding["input_ids"].clone()
        
        # 添加样本ID用于后续追踪
        encoding["id"] = sample["json_key"]
        
        if self.mode == 'predict':
            encoding["instruction_text"] = sample["instruction_only"]
        
        return encoding
    
if __name__ == "__main__":
    print("debug")

    tokenizer = AutoTokenizer.from_pretrained("../Qwen/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = InsDataset("../data/dev.json", tokenizer, max_length=1024, mode='train')
    
    print("=" * 50)
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor shape {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    preview = text[:1024]
    print(preview)
    