import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import torch

MODEL_NAME = "./qwen25-coder-1b-instruct"
DATA_PATH  = "train.jsonl"           # your dataset
OUT_DIR    = "qwen25-1b-dsl-qlora"
MAX_LEN    = 128

class CudaPeakMemoryCallback(TrainerCallback):
    def __init__(self):
        self.max_allocated = 0
        self.max_reserved  = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # for multi-GPU, reset each device
            for d in range(torch.cuda.device_count()):
                with torch.cuda.device(d):
                    torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        # track per-device peaks
        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):
                self.max_allocated = max(self.max_allocated, torch.cuda.max_memory_allocated())
                self.max_reserved  = max(self.max_reserved,  torch.cuda.max_memory_reserved())

    def on_train_end(self, args, state, control, **kwargs):
        def gb(x): return x / (1024**3)
        print(f"\n=== GPU memory (PyTorch) ===")
        print(f"Peak allocated: {gb(self.max_allocated):.2f} GB")
        print(f"Peak reserved : {gb(self.max_reserved):.2f} GB\n")

# 4-bit QLoRA config (memory friendly)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"  # set to "float16" if your GPU lacks bf16
)

# LoRA config (lean for small VRAM; you can raise r later)
lora_cfg = LoraConfig(
    r=8,                   # start small; bump to 16 if VRAM allows
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

# Training args tuned for ~14 GB VRAM
sft_cfg = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,     # safe; increase to 3–4 if you have headroom
    gradient_accumulation_steps=8,     # effective batch = 16
    learning_rate=2e-4,                # LoRA likes higher LR than full-FT
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=20,
    save_strategy="epoch",
    bf16=True,                         # switch to fp16=True if needed
    fp16=False,
    packing=True,                      # packs multiple short samples into one sequence
    max_seq_length=MAX_LEN,
    gradient_checkpointing=True,       # huge memory saver
    optim="paged_adamw_8bit"           # memory-friendly optimizer
)

ds = load_dataset("json", data_files={"train": DATA_PATH})

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map="auto",                 # single GPU auto-maps
)

callbacks = [CudaPeakMemoryCallback()]

trainer = SFTTrainer(
    model=model,
    #tokenizer=tok,
    train_dataset=ds["train"],
    peft_config=lora_cfg,
    args=sft_cfg,
    callbacks=callbacks,
)

trainer.train()
trainer.save_model()
tok.save_pretrained(OUT_DIR)
print("Saved ->", OUT_DIR)
