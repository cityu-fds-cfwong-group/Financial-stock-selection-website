import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

dataset = load_dataset("json", data_files="train.json")

model_id = "Qwen/Qwen3-4B-thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def preprocess_function(examples):
    prompts = []
    for user, assistant in zip(examples["user"], examples["assistant"]):
        prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"
        prompts.append(prompt)
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    tokenized["labels"] = tokenized["input_ids"].clone()
    tokenized["labels"][tokenized["attention_mask"] == 0] = -100
    
    return tokenized

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

output_dir = "./qwen3_4b_finance_lora"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_8bit",
    remove_unused_columns=False
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)
trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"LoRa模型已保存至：{output_dir}")