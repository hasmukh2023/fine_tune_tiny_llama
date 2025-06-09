import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# Load dataset
def load_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list([{"text": f"<s>[INST] {d['prompt']} [/INST] {d['response']} </s>"} for d in data])

dataset = load_jsonl("hasmukh_data.jsonl")

# Model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize)

# Load model (on CPU)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

# Apply LoRA
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.1, bias="none")
model = get_peft_model(model, peft_config)

# Training config
args = TrainingArguments(
    output_dir="./finetuned_cpu",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=1e-4,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained("./finetuned_cpu")
tokenizer.save_pretrained("./finetuned_cpu")