"""
Step 4: Attach LoRA Adapter & Inspect Trainable Parameters
Demonstrates how LoRA freezes base model weights and only trains
a tiny fraction (<1%) of parameters.
"""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import MODEL_ID, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES

def main():
    print("=" * 60)
    print("STEP 4: LoRA Adapter Setup & Parameter Inspection")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required to prepare model for kbit LoRA training.")
        return

    # 1. 4-bit configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading 4-bit base model: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    # 2. Prepare k-bit model for training
    # Note: Default PEFT prepare_model_for_kbit_training attempts to upcast ALL non-4bit
    # parameters to float32. For Gemma 4, that tries to upcast the 2.35B embedding table
    # requiring 8.75 GiB VRAM and causes OOM. We freeze base weights and cast only norms to float32!
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "norm" in name and param.__class__.__name__ != "Params4bit":
            param.data = param.data.to(torch.float32)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("\nAttaching LoRA adapter...")
    model = get_peft_model(model, peft_config)

    print("\n--- Trainable Parameters Summary ---")
    model.print_trainable_parameters()

    print("\n--- Target Modules Adapted ---")
    for mod in LORA_TARGET_MODULES:
        print(f"  [+] {mod}")

    print("\nNotice how only a tiny fraction of weights (~0.5%) will be updated.")
    print("All base model weights remain frozen in 4-bit.")
    print("=" * 60)
    print("[SUCCESS] LoRA adapter is verified and ready for training!")
    print("=" * 60)

if __name__ == "__main__":
    main()
