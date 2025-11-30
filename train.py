import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from transformers import TrainingArguments, Trainer, AutoTokenizer
import numpy as np
from data_loader import HomonymDataset, collate_fn
from model import DebertaRationalityScorer

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    # 确保预测值在1-5范围内
    predictions = np.clip(predictions, 1, 5)
    
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    
    # 计算准确率（误差在0.5分以内）
    accuracy = np.mean(np.abs(predictions - labels) <= 0.5)
    
    return {
        "mse": mse,
        "mae": mae, 
        "accuracy_0.5": accuracy
    }

def train_model():
    """训练模型"""
    
    # 配置
    model_name = "./deberta-v3-large"
    data_path = "./data/train.json"  # 根据你的数据路径调整
    output_dir = "./deberta-regression-model"
    batch_size = 8
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = DebertaRationalityScorer(model_name)
    
    # 加载数据
    dataset = HomonymDataset(data_path, tokenizer)
    
    # 分割数据集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        learning_rate=2e-5,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        # remove_unused_columns=False,
    )
    
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"训练完成，模型保存在: {output_dir}")
    return trainer

if __name__ == "__main__":
    train_model()