import json
import requests
import os
from pathlib import Path
from tqdm import tqdm

API_URL = "http://localhost:8080/completion"
JSON_PATH = Path("../data/dev.json").resolve()
OUTPUT_PATH = Path("../results/predictions.jsonl").resolve()

def build_prompt(item):
    """构建与之前完全一致的Prompt"""
    context = f"{item['precontext']}\n{item['sentence']}\n{item['ending']}"
    
    instruction = f"""You are an expert in semantic disambiguation. 
Please analyze the homonym word '{item['homonym']}' within the given context.
Determine how likely this word is to carry the meaning: '{item['judged_meaning']}'.

Output ONLY a single integer from 1 to 5 where:
1 = Very Unlikely
2 = Unlikely  
3 = Neutral
4 = Likely
5 = Very Likely

Do not output any other text.
### example respense:
4
"""


    input_text = f"""Homonym: {item['homonym']}
Judged Meaning: {item['judged_meaning']}
Example Sentence: {item['example_sentence']}

Context:
{context}"""
    # 下面全部拼起来才是 return 的内容
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

def get_prediction(prompt):
    """通过 API 获取推理结果"""
    payload = {
        "prompt": prompt,
        "n_predict": 8,
        "temperature": 0.75,
        "stop": ["\n", "###", "Input:", "Instruction:"],
        "trim_prompt": True
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["content"].strip()
    except Exception as e:
        print(f"请求失败: {e}")
        return ""

def parse_score(text):
    """提取第一个出现的 1-5 数字"""
    for char in text:
        if char in '12345':
            return int(char)
    return -1

def main():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.items() if isinstance(data, dict) else enumerate(data)
    
    print(f"开始处理 {len(data)} 条数据...")
    

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        for key, item in tqdm(items):
            prompt = build_prompt(item)
            raw_output = get_prediction(prompt)
            score = parse_score(raw_output)
            
            result = {
                "id": key,
                "prediction": score,
                # "raw_response": raw_output
            }
            
            # 写入 jsonl
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

    print(f"\n任务完成！结果已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()