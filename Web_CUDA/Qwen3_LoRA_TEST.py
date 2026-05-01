import json
import os
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import ast
import random

DATA_FOLDER = "CFLUE" 
TEST_FILENAME = "test.json"
LORA_PATH = "./qwen3_4b_finance_lora"
BASE_MODEL_PATH = "Qwen/Qwen3-4B-Thinking-2507" 

# 设置为 0.1 表示只跑 10% 的数据，设置为 1.0 表示跑全部数据
EVAL_RATIO = 0.1


# 是否打乱数据。True: 先随机打乱再截取。False: 直接截取前 N 条
USE_SHUFFLE = False 

# 固定随机种子，保证每次抽样结果一致，方便复现
RANDOM_SEED = 42 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(SCRIPT_DIR, DATA_FOLDER, TEST_FILENAME)
LORA_FULL_PATH = os.path.join(SCRIPT_DIR, LORA_PATH)

def check_paths():
    if not os.path.exists(TEST_FILE):
        print(f"❌ 错误：找不到测试集 -> {TEST_FILE}")
        return False
    if not os.path.exists(LORA_FULL_PATH):
        print(f"❌ 错误：找不到 LoRA 文件夹 -> {LORA_FULL_PATH}")
        return False
    if not os.path.exists(os.path.join(LORA_FULL_PATH, "adapter_config.json")):
        print(f"❌ 错误：LoRA 文件夹中缺少 'adapter_config.json'")
        return False
    
    print(f"   路径检查通过")
    print(f"   测试集：{TEST_FILE}")
    print(f"   LoRA路径：{LORA_FULL_PATH}")
    return True

def parse_choices(choices_str):
    if not choices_str:
        return {}
    try:
        return ast.literal_eval(choices_str)
    except Exception:
        return {}

def format_prompt_for_qwen_thinking(sample):
    question_text = sample.get('question', '')
    choices_str = sample.get('choices', '')
    task_name = sample.get('名称', '金融测试')
    
    choices_dict = parse_choices(choices_str)
    options_text = ""
    if choices_dict:
        for key, value in choices_dict.items():
            options_text += f"{key}. {value}\n"
    else:
        options_text = choices_str

    system_prompt = "你是一个专业的金融考试助手。请阅读题目和选项，进行逐步推理分析，最后给出正确选项的字母（A/B/C/D）。"
    
    user_content = f"""【科目】{task_name}
【题目】{question_text}
【选项】
{options_text}
请逐步思考并给出最终答案（只需输出选项字母）："""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    return messages

def extract_answer_letter(text):
    text = text.upper()
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1)
    return text.strip()

def evaluate_model(model, tokenizer, dataset):
    predictions = []
    references = []
    correct_count = 0
    
    total_samples = len(dataset)
    print(f"开始评测，共 {total_samples} 条数据...")
    
    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        true_answer = str(sample.get('answer', '')).strip().upper()
        messages = format_prompt_for_qwen_thinking(sample)
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,       
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        processed_pred = extract_answer_letter(raw_response)
        
        predictions.append(processed_pred)
        references.append(true_answer)
        
        if processed_pred == true_answer:
            correct_count += 1

        if idx < 3:
            print("\n" + "="*30)
            print(f"[样例 {idx+1}]")
            print(f"题目: {sample.get('question', '')[:50]}...")
            print(f"真实答案: {true_answer}")
            print(f"模型提取结果: {processed_pred} -> {'✅ 正确' if processed_pred == true_answer else '❌ 错误'}")
            print("="*30 + "\n")

    return predictions, references, correct_count

def main():
    if not check_paths():
        return

    print(f"正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    print(f"正在加载基座模型：{BASE_MODEL_PATH} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()

    print(f"正在挂载 LoRA 适配器...")
    try:
        model = PeftModel.from_pretrained(base_model, LORA_FULL_PATH)
        model = model.merge_and_unload()
        print("模型准备就绪")
    except Exception as e:
        print(f"❌ LoRA 挂载失败：{e}")
        return

    print(f"正在加载数据...")
    full_dataset = load_dataset("json", data_files={"test": TEST_FILE})["test"]
    total_len = len(full_dataset)
    
    if EVAL_RATIO >= 1.0:
        eval_dataset = full_dataset
        print(f"设置比例为 {EVAL_RATIO}，将运行全部 {total_len} 条数据。")
    else:
        num_samples = int(total_len * EVAL_RATIO)
        print(f"原始数据量：{total_len} 条")
        print(f"设定比例：{EVAL_RATIO} -> 即将评测 {num_samples} 条数据")
        
        if USE_SHUFFLE:
            print(f"正在打乱数据顺序 (Seed={RANDOM_SEED})...")
            generator = torch.Generator()
            generator.manual_seed(RANDOM_SEED)
            eval_dataset = full_dataset.shuffle(seed=RANDOM_SEED).select(range(num_samples))
        else:
            print(f"直接截取前 {num_samples} 条数据...")
            eval_dataset = full_dataset.select(range(num_samples))
            
        print(f"数据集采样完成，当前大小：{len(eval_dataset)}")

    preds, refs, correct_count = evaluate_model(model, tokenizer, eval_dataset)

    accuracy = correct_count / len(refs) if len(refs) > 0 else 0
    
    print("\n" + "#"*40)
    print(f"抽样评测完成！")
    print(f"评测样本数：{len(refs)} (占总集 {total_len} 的 {len(refs)/total_len:.2%})")
    print(f"答对数量：{correct_count}")
    print(f"估算准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    if USE_SHUFFLE:
        print("提示：由于开启了随机打乱，此准确率具有较好的代表性。")
    else:
        print("提示：未开启随机打乱，此结果仅反映前部分数据的性能。")
    print("#"*40)

if __name__ == "__main__":
    main()