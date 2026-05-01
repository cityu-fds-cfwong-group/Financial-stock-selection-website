import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def main():
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    lora_relative_path = "qwen3_4b_finance_lora"
    
    output_filename = "qwen3_response.txt"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    lora_path = os.path.join(script_dir, lora_relative_path)
    output_file_path = os.path.join(script_dir, output_filename)

    print(f"当前脚本所在目录: {script_dir}")
    print(f"尝试加载 LoRA 的完整路径: {lora_path}")
    
    if not os.path.exists(lora_path):
        print(f"错误: 找不到 LoRA 文件夹 '{lora_path}'")
        print("请确保 'qwen3_4b_finance_lora' 文件夹与脚本在同一目录下。")
        return
    
    if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        print(f"错误: 在 '{lora_path}' 中未找到 'adapter_config.json'")
        return
        
    print(f"结果将保存到: {output_file_path}")
    print("-" * 30)

    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("正在加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print(f"正在加载 LoRA 适配器从: {lora_path} ...")
    try:
        model = PeftModel.from_pretrained(model, lora_path)
        print("LoRA 加载成功！")
    except Exception as e:
        print(f"加载 LoRA 失败: {e}")
        return

    model.eval()

    print("模型加载完成，开始对话。输入 'quit' 退出。")
    
    while True:
        try:
            user_input = input("\n请输入问题 (输入 'quit' 退出): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n收到中断信号，安全退出。")
            break
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n 正在退出程序...")
            break
        
        if not user_input:
            continue

        messages = [
            {"role": "system", "content": "你是一个专业的金融助手。"},
            {"role": "user", "content": user_input}
        ]
        
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<|im_start|>system\n你是一个专业的金融助手。<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print(f"\n模型回答:\n{response}")

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(f"提问:\n{user_input}\n\n")
                f.write(f"模型回答:\n{response}\n")
            print(f"\n[系统提示] 本次问答已保存至: {output_file_path}")
        except Exception as e:
            print(f"\n[系统警告] 保存文件失败: {e}")

if __name__ == "__main__":
    main()