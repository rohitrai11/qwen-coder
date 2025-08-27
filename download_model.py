import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Pick your model
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SAVE_PATH = "./qwen25-coder-1b-instruct"   # local folder

# 1) Create folder if not exists
os.makedirs(SAVE_PATH, exist_ok=True)

# 2) Download and save tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.save_pretrained(SAVE_PATH)

# 3) Download and save model weights
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",   # or torch.float16 / torch.bfloat16
    device_map="cpu"      # keep on CPU while saving
)
model.save_pretrained(SAVE_PATH)

print(f"✅ Model and tokenizer saved to {os.path.abspath(SAVE_PATH)}")
