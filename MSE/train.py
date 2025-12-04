import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import DebertaV2ForWSDScoring
from data_load import MSEDataset

from transformers import DebertaV2Tokenizer, DebertaV2Config, Trainer, TrainingArguments
# 假设您的模型和数据集类定义在 model.py 和 data_load.py 中
# from model import DebertaV2ForWSDScoring  # 您的回归模型
# from data_load import MSEDataset          # 您的回归数据集（原SimpleWSDDataset修改版）

# 建议使用相对较小的版本开始，以节省资源
MODEL_NAME = "microsoft/deberta-v3-large" 
TRAIN_JSON_PATH = "../data/train.json" # 假设您的数据路径

# 1. 加载 Tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

# 2. 加载配置（用于初始化您的模型类）
config = DebertaV2Config.from_pretrained(MODEL_NAME)

# **【关键修改】**：
# 对于回归任务，模型的输出类别数量 (NUM_OUTPUTS) 应该是 1。
# 您的自定义模型 DebertaV2ForWSDScoring 的 __init__ 方法应该使用这个配置。
config.num_labels = 1 

# 3. 初始化您的自定义模型
# 请确保您的 DebertaV2ForWSDScoring 类已经修改为输出 1 个值并使用 nn.MSELoss
# 假设您已在脚本中导入了 DebertaV2ForWSDScoring 类
model = DebertaV2ForWSDScoring.from_pretrained(
    MODEL_NAME, 
    config=config
)

# 导入您的数据集类 (我们称之为 MSEDataset 或使用您修改后的 SimpleWSDDataset)
train_dataset = MSEDataset( # 假设这是您修改后输出 float labels (avg) 的类
    json_path=TRAIN_JSON_PATH, 
    tokenizer=tokenizer
)

OUTPUT_DIR = "./output_regression"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,                     
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    # ------------------------------------------------
    # 禁用评估
    eval_strategy="no", 
    load_best_model_at_end=False, 
    # ------------------------------------------------
    warmup_steps=500,                       
    weight_decay=0.01,                      
    logging_dir='./logs_regression',       
    logging_steps=50,                       
    save_strategy="no",                  # no
    learning_rate=2e-5,                     
    fp16=True,                              # 混合精度训练，加速
    # **【可选优化】**：报告指标为回归任务
    # metric_for_best_model="eval_loss",      # 尽管我们禁用了评估，但保留此设置
    # greater_is_better=False,
    seed=42,                                # 固定的随机种子
)

# 实例化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    # **【回归任务不需要 special data collator】**：
    # Data collator 默认会处理回归任务的 float labels
)

# 启动训练
print("开始微调 DeBERTaV2 回归模型...")
trainer.train()

# 训练结束后，保存最终模型
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成，模型和tokenizer已保存至 {OUTPUT_DIR}")