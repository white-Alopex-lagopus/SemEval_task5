import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from data_load import InsDataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="../Qwen/Qwen3-4B",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = InsDataset(
        json_path="../data/train.json",
        tokenizer=tokenizer,
        max_length=1024,
        mode='train'
    )
    eval_dataset = InsDataset(
        json_path="../data/dev.json",
        tokenizer=tokenizer,
        max_length=1024,
        mode='eval'
    )
    print(f"数据: {len(train_dataset)} 训练, {len(eval_dataset)} 验证")

    # 3. 加LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    )

    # 4. 训练
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="full_text",
        max_seq_length=1024,
        args=TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            num_train_epochs=3,
            learning_rate=2e-5,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_steps=200,
            eval_strategy="epoch",
            eval_steps=50,
            save_strategy="no",
            load_best_model_at_end=False,
        ),
    )

    trainer.train()
    trainer.save_model()
    print("训练完成，模型已保存")

if __name__ == "__main__":
    main()