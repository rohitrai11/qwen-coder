# infer_qwen25_coder.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

CKPT = "./qwen25-coder-dsl-fullft/checkpoint-54"  # or full-FT dir

tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(CKPT, device_map="auto", torch_dtype=torch.bfloat16)

SYSTEM = "You translate English math tasks into valid algebra DSL code. Only output code."
STOPS  = ["\n\n"]   # helps stop after first block

def generate_code(instruction: str, max_new_tokens=96):
    messages = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":instruction}
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,          # deterministic
        top_p=1.0,
        do_sample=False,
        eos_token_id=tok.eos_token_id
    )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # crude stop at first blank line if present
    text = text.split("\n\n")[0].strip()
    return text

print(generate_code("Define a polynomial ring over GF(7)."))
