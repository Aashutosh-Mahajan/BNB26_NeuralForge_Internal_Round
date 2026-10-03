"""QLoRA fine-tune of Qwen2.5-1.5B-Instruct as the Black Box explainer (FR-16, P2).

Run on a free Kaggle T4 (GPU: T4 x1, Internet: on), not on the laptop:
  1. Upload data/sft_explainer.jsonl as a Kaggle dataset.
  2. !pip install -q "transformers>=4.45" "peft>=0.13" "trl>=0.11" "bitsandbytes>=0.44" accelerate datasets
  3. !python kaggle_qlora_explainer.py --data /kaggle/input/<dataset>/sft_explainer.jsonl
  4. Download /kaggle/working/blackbox-explainer and set EXPLAINER_ADAPTER_PATH in .env.

~1-1.5 h on a T4 for 1,500 examples x 2 epochs.
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="/kaggle/working/blackbox-explainer")
    parser.add_argument("--epochs", type=float, default=2)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, device_map="auto", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    data = load_dataset("json", data_files=args.data, split="train").train_test_split(test_size=0.05, seed=42)
    config = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=4,
                       gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine",
                       warmup_ratio=0.03, logging_steps=20, eval_strategy="epoch", save_strategy="epoch",
                       fp16=True, max_seq_length=2048, report_to="none")
    trainer = SFTTrainer(model=model, args=config, train_dataset=data["train"], eval_dataset=data["test"],
                         peft_config=lora, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print("Saved adapter to", args.out)


if __name__ == "__main__":
    main()
