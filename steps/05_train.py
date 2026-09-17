"""
Step 5: Run QLoRA Fine-Tuning
Trains the LoRA adapter on dataset_train.jsonl and saves
the final adapter weights to ./gemma-router-qlora/final_adapter.
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from config import (
    MODEL_ID, DATA_TRAIN, DATA_VAL, ADAPTER_OUTPUT_DIR, FINAL_ADAPTER_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, NUM_EPOCHS, MAX_SEQ_LENGTH
)

def main():
    print("=" * 60)
    print("STEP 5: Starting QLoRA Fine-Tuning")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] GPU is required for QLoRA training.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Tokenizer & Base Model
    print(f"Loading {MODEL_ID} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Load Dataset
    print(f"\nLoading dataset from {DATA_TRAIN} and {DATA_VAL}...")
    dataset = load_dataset("json", data_files={
        "train": str(DATA_TRAIN),
        "validation": str(DATA_VAL)
    })

    def format_prompts(batch):
        formatted_texts = []
        for conversation in batch["messages"]:
            text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
        return {"text": formatted_texts}

    train_ds = dataset["train"].map(format_prompts, batched=True)
    val_ds = dataset["validation"].map(format_prompts, batched=True)

    # 5. Training Arguments (Tuned for 8GB VRAM)
    from trl import SFTConfig
    training_args = SFTConfig(
        output_dir=str(ADAPTER_OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        warmup_steps=10,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH
    )

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args
    )

    print("\nTraining starts now...")
    trainer.train()

    # 7. Save Final Adapter
    print(f"\nSaving final adapter to: {FINAL_ADAPTER_DIR}...")
    trainer.model.save_pretrained(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))

    print("=" * 60)
    print(f"[SUCCESS] QLoRA training complete! Adapter saved to:\n  {FINAL_ADAPTER_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
