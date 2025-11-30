import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class HomonymDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.examples = self._preprocess_data()
        print(f"数据集初始化完成，共有 {len(self.examples)} 个样本")
    
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
                "item_id": int(item_id),
                "raw_data": item
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
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(example["score"], dtype=torch.float),
            "item_id": example["item_id"]
        }
        
        # 调试：检查返回的键
        print(f"__getitem__({idx}) 返回的键: {list(result.keys())}")
        return result

def collate_fn(batch):
    """自定义批处理函数"""
    print(f"collate_fn 接收到 {len(batch)} 个样本")
    
    # 检查每个样本的键
    for i, item in enumerate(batch):
        print(f"  样本 {i} 的键: {list(item.keys())}")
        if 'item_id' not in item:
            print(f"  ⚠️ 样本 {i} 缺少 'item_id' 键!")
    
    try:
        result = {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "item_ids": [item["item_id"] for item in batch]
        }
        print("collate_fn 执行成功")
        return result
    except KeyError as e:
        print(f"❌ collate_fn 错误: {e}")
        print(f"批处理中第一个样本的键: {list(batch[0].keys())}")
        raise

def debug_training_issue():
    """调试训练过程中的问题"""
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./deberta-v3-large")
    
    # 创建数据集
    dataset = HomonymDataset("/home/baimingyu/workspace/SemEval/data/train.json", tokenizer)
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=collate_fn,
        shuffle=True
    )
    
    print("\n" + "="*80)
    print("开始测试DataLoader迭代...")
    print("="*80)
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n✅ 成功处理第 {batch_idx} 个batch")
            print(f"   batch keys: {list(batch.keys())}")
            print(f"   item_ids: {batch['item_ids']}")
            
            if batch_idx >= 2:  # 只测试前3个batch
                break
                
    except Exception as e:
        print(f"\n❌ 迭代过程中出错: {e}")
        import traceback
        traceback.print_exc()

def check_for_parallel_issues():
    """检查是否与并行处理相关的问题"""
    print("\n" + "="*80)
    print("检查num_workers问题...")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained("./deberta-v3-large")
    dataset = HomonymDataset("/home/baimingyu/workspace/SemEval/data/train.json", tokenizer)
    
    # 测试不同的num_workers设置
    for num_workers in [0, 1, 2]:
        print(f"\n测试 num_workers = {num_workers}")
        try:
            dataloader = DataLoader(
                dataset, 
                batch_size=2, 
                collate_fn=collate_fn,
                num_workers=num_workers
            )
            
            batch = next(iter(dataloader))
            print(f"  ✅ num_workers={num_workers} 工作正常")
            
        except Exception as e:
            print(f"  ❌ num_workers={num_workers} 出错: {e}")

if __name__ == "__main__":
    # 运行调试
    debug_training_issue()
    check_for_parallel_issues()