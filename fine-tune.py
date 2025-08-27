# train_full_qwen25_coder.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "./qwen25-coder-1b-instruct"
DATA_PATH  = "train.jsonl"
OUT_DIR    = "qwen25-coder-dsl-fullft"
MAX_LEN    = 512

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="bfloat16",
    device_map="auto"
)

cfg = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=4,      # raise if VRAM allows
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=20,
    save_strategy="epoch",
    bf16=True,
    packing=True,
    max_seq_length=MAX_LEN,
    gradient_checkpointing=True
)

ds = load_dataset("json", data_files={"train": DATA_PATH})

trainer = SFTTrainer(
    model=model,
    #tokenizer=tok,
    train_dataset=ds["train"],
    args=cfg,
)
trainer.train()
trainer.save_model()
tok.save_pretrained(OUT_DIR)
print(f"✅ Model and tokenizer saved to {OUT_DIR}")