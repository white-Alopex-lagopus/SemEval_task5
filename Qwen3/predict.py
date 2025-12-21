import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from data_load import InsDataset

def main():
    BASE_MODEL = "../Qwen/Qwen3-4B"
    LORA_PATH = "./output"
    TEST_DATA_PATH = "../data/dev.json"
    OUTPUT_FILE = "predictions.jsonl"
    
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("创建测试数据集...")
    test_dataset = InsDataset(
        json_path=TEST_DATA_PATH,
        tokenizer=tokenizer,
        max_length=1024,
        mode='predict'
    )
    
    print(f"加载了 {len(test_dataset)} 个测试样本")
    
    prompts = []
    sample_keys = []
    for i in range(len(test_dataset)):
        sample = test_dataset.samples[i]
        prompts.append(sample["full_text"])
        sample_keys.append(sample["json_key"])
    
    print("加载模型和LoRA适配器...")
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=1024,
    )
    
    lora_request = LoRARequest(
        lora_name="trained_lora",
        lora_int_id=1,
        lora_local_path=LORA_PATH,
    )
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=10,
        stop=["<|im_end|>", "\n"]
    )
    
    print("开始推理...")
    batch_size = 8
    all_results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_keys = sample_keys[i:i+batch_size]
        
        outputs = llm.generate(
            batch_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            
            score = 0  # 默认值
            for char in generated_text:
                if char.isdigit():
                    score = int(char)
                    # 确保分数在1-5范围内
                    if 1 <= score <= 5:
                        break
                    else:
                        score = 4
                        break
        
            all_results.append({
                "id": batch_keys[j],
                "prediction": score,
                "generated_text": generated_text
            })
        
        processed = min(i+batch_size, len(prompts))
        print(f"已处理 {processed}/{len(prompts)} 个样本")
 
    print(f"\n保存结果到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in all_results:

            json_line = {
                "id": str(result["id"]),
                "prediction": int(result["prediction"])
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    
    print(f"推理完成！结果已保存到 {OUTPUT_FILE}")
    
    valid_predictions = [r for r in all_results if 1 <= r["prediction"] <= 5]
    print(f"\n统计信息:")
    print(f"总样本数: {len(all_results)}")
    print(f"有效预测(1-5分): {len(valid_predictions)}")
    print(f"无效预测(0分): {len(all_results) - len(valid_predictions)}")
    
    if valid_predictions:
        scores = [r["prediction"] for r in valid_predictions]
        print("分数分布:")
        for s in range(1, 6):
            count = scores.count(s)
            percentage = count / len(valid_predictions) * 100
            print(f"  {s}分: {count:3d} 个 ({percentage:.1f}%)")

    print("\n前5个预测结果:")
    for i, result in enumerate(all_results[:5]):
        print(f"{i+1}. ID: {result['id']}")
        print(f"   模型输出: {result['generated_text']}")
        print(f"   预测分数: {result['prediction']}")
        print()

if __name__ == "__main__":
    main()