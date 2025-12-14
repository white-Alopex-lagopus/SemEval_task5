from transformers import DebertaV2Tokenizer
from data_load import ORDataset
from torch.utils.data import DataLoader

tokenizer = DebertaV2Tokenizer.from_pretrained('I:\deberta-v3-large')

dev_dataset = ORDataset(
    json_path="../data/dev.json", 
    tokenizer=tokenizer, 
    mode='train',
)

print(f"数据集大小: {len(dev_dataset)}")

# 通过 DataLoader 迭代获取数据
dataloader = DataLoader(dev_dataset, batch_size=2)

# 打印出 batch 数据
for batch in dataloader:
    print(batch)
    break  # 只打印一个批次数据