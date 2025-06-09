import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# Make sure model runs on CPU
device = "cpu"
model.to(device)

# Define a sample prompt
prompt = "<s>[INST] Who is hasmukh mer? [/INST]"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
