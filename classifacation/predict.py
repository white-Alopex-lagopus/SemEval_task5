import torch
import os
import json
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer, AutoConfig

# ⚠️ 确保导入了您的自定义类
from model import DebertaV2ForWSDScoring 
from data_load import SimpleWSDDataset 
# 注意：SimpleWSDDataset 默认会跳过 'nonsensical' 为 True 的样本。
# 如果 dev.json 中包含 nonsensical 样本且需要评分，你需要一个不同的 Dataset 类。
# 假设 dev.json 只包含需要评分的样本，且我们只关心评分结果。


# --- 配置 ---
OUTPUT_DIR = "./infer"
DEV_JSON_PATH = "../data/dev.json"  # 假设 dev.json 路径
RESULT_JSONL_PATH = "./dev_predictions.jsonl"
BATCH_SIZE = 32  # 可以根据您的 GPU 内存调整
NUM_LABELS = 5   # 1-5分
MAX_LENGTH = 512

# --- 1. 设置设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 模型推理将在设备上运行: {device}")

# --- 2. 加载模型和 Tokenizer ---
try:
    print(f"正在从 {OUTPUT_DIR} 加载模型和 Tokenizer...")
    config = AutoConfig.from_pretrained(OUTPUT_DIR)
    tokenizer = DebertaV2Tokenizer.from_pretrained(OUTPUT_DIR)
    
    model = DebertaV2ForWSDScoring.from_pretrained(OUTPUT_DIR, config=config)
    
    model.to(device)
    model.eval()
    print("✅ 模型和 Tokenizer 加载成功。")

except Exception as e:
    print(f"❌ 加载失败。请检查 {OUTPUT_DIR} 中的文件是否完整，以及 'model.py' 和 'data_load.py' 是否正确。")
    print(f"错误信息: {e}")
    exit()

# --- 3. 加载数据集和 DataLoader ---
# ⚠️ 注意: SimpleWSDDataset 默认会读取 JSON 中的 score 作为 labels。
# 对于推理，我们只是用它来生成 input_ids，labels 会被忽略。
print(f"正在加载 {DEV_JSON_PATH} 数据集...")
test_dataset = SimpleWSDDataset(
    json_path=DEV_JSON_PATH, 
    tokenizer=tokenizer, 
    max_length=MAX_LENGTH
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    # Tokenizer 已经进行了 padding，这里不需要 collate_fn
    collate_fn=None 
)
print(f"共生成 {len(test_dataset)} 个推理样本，分 {len(test_dataloader)} 批次处理。")

# --- 4. 执行批量推理 ---
all_predictions = []
all_sample_ids = []

print("--- 开始批量推理 ---")
with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
        # 4.1 准备输入
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # 4.2 运行模型
        # model.forward 只返回 logits，因为 batch 中没有 labels
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # logits.shape: [batch_size, 5] (5个类别)

        # 4.3 后处理：找出预测的 1-5 分
        # 找到 5 个类别中概率最高的索引 (0-4)
        predicted_scores_index = torch.argmax(logits, dim=1)
        # 转换为 1-5 分
        predicted_scores = predicted_scores_index + 1
        
        all_predictions.extend(predicted_scores.cpu().numpy())
        
        # 4.4 收集样本 ID
        # ⚠️ WARNING: DataLoader 返回的 batch 中不包含 SimpleWSDDataset 中的 'sample_id'。
        # 由于 SimpleWSDDataset 是基于索引的，我们必须手动映射 ID。
        start_idx = step * BATCH_SIZE
        end_idx = min((step + 1) * BATCH_SIZE, len(test_dataset))
        
        # 从 dataset 对象的 samples 列表中提取 sample_id
        current_sample_ids = [
            test_dataset.samples[i]['sample_id'] 
            for i in range(start_idx, end_idx)
        ]
        all_sample_ids.extend(current_sample_ids)
        
        if (step + 1) % 50 == 0:
            print(f"已处理 {step + 1}/{len(test_dataloader)} 批次...")

print("✅ 推理完成。")

# --- 5. 保存结果到 JSONL 文件 ---

print(f"正在保存结果到 {RESULT_JSONL_PATH}...")
output_records = []

# 5.1 构造 JSONL 记录
for sample_id, prediction in zip(all_sample_ids, all_predictions):
    # SimpleWSDDataset 的 sample_id 是 "original_id_choice_idx"
    # 我们按照题目的要求，保存这个展平后的样本 ID 和预测得分
    output_records.append({
        "id": str(sample_id),
        "prediction": int(prediction) # 预测的 1-5 分
    })

# 5.2 写入 JSONL 文件
with open(RESULT_JSONL_PATH, 'w', encoding='utf-8') as f:
    for record in output_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"🎉 结果成功保存到 {RESULT_JSONL_PATH}。共 {len(output_records)} 条记录。")

# 示例输出格式检查：
# {"id": "sample_123_0", "prediction": 5}