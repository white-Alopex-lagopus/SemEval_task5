from transformers import DebertaV2Tokenizer, DebertaV2Config, Trainer, TrainingArguments

from model import DebertaV2ForOrdinalRegression
from data_load import ORDataset



# --- 配置 ---
MODEL_NAME = "microsoft/deberta-v3-large" 
TRAIN_JSON_PATH = "../data/train.json" 
EVAL_JSON_PATH = "../data/dev.json" # 使用 dev.json 进行验证
OUTPUT_DIR = "./output_ordinal_regression"

# ----------------------------------------------------------------------
# 2. 主训练流程
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 加载 Tokenizer 和配置
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    config = DebertaV2Config.from_pretrained(MODEL_NAME)
    
    # 实例化序数回归模型
    # 注意：这里不再需要 num_labels=1，因为模型内部处理了输出维度
    model = DebertaV2ForOrdinalRegression.from_pretrained(MODEL_NAME, config=config)
    
    # 实例化数据集
    train_dataset = ORDataset(json_path=TRAIN_JSON_PATH, tokenizer=tokenizer, mode='train')

    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                       # 增加 epochs
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        
        save_strategy="no",                    # no
        
        warmup_steps=500,                       
        weight_decay=0.01,                      
        logging_dir='./logs_ordinal',       
        logging_steps=50,                       
        learning_rate=1e-5,                     
        fp16=True,                              
        seed=42,
        remove_unused_columns=False,                   
    )

    # 实例化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # 启动训练
    print("***** 开始微调 DeBERTaV2 序数回归模型 *****")
    trainer.train()

    # 训练结束后，保存最终模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"训练完成，最终最佳模型已保存至 {OUTPUT_DIR}")