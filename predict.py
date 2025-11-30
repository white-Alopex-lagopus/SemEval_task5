import os
import json
import torch
from transformers import AutoTokenizer
from model import DebertaRationalityScorer

class Predictor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = DebertaRationalityScorer.from_pretrained(model_path)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def predict_single(self, text: str) -> float:
        """预测单个文本的合理性分数"""
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = outputs.logits.squeeze().cpu().item()
        
        return prediction
    
    def predict_batch(self, data_path: str, output_path: str):
        """批量预测并生成评估格式的预测文件"""
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        
        predictions = []
        
        for item_id, item in gold_data.items():
            # 构建输入文本
            input_text = f"{item['precontext']} {item['sentence']} {item['ending']}".strip()
            
            # 预测
            score = self.predict_single(input_text)
            
            predictions.append({
                "id": int(item_id),  # 转换为整数以匹配评估脚本
                "prediction": score
            })
        
        # 保存预测结果
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
        
        print(f"预测完成，结果保存至: {output_path}")
        return predictions

def main():
    # 使用训练好的模型进行预测
    model_path = "./deberta-regression-model"  # 训练好的模型路径
    data_to_predict = "data/dev.json"  # 要预测的数据
    output_file = "predictions/deberta_predictions.jsonl"  # 输出文件
    
    # 创建输出目录
    os.makedirs("predictions", exist_ok=True)
    
    predictor = Predictor(model_path)
    predictions = predictor.predict_batch(data_to_predict, output_file)
    
    print(f"生成了 {len(predictions)} 个预测")
    print("前5个预测样例:")
    for pred in predictions[:5]:
        print(f"ID: {pred['id']}, Prediction: {pred['prediction']:.3f}")

if __name__ == "__main__":
    main()