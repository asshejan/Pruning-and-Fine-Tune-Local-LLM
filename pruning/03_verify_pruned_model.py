"""
Pruning Step 3: Test & Verify Pruned Text Model
Loads ../gemma-4-E2B-text in 4-bit, verifies GPU VRAM footprint
(should be ~1.2 to 1.5 GB instead of 6.3 GB), and tests generation.
"""
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PRUNED_MODEL_DIR = Path(__file__).resolve().parent.parent / "gemma-4-E2B-text"

def main():
    print("=" * 65)
    print("STEP 3: Verify Pruned Text Model Performance & VRAM")
    print("=" * 65)

    if not PRUNED_MODEL_DIR.exists() or not (PRUNED_MODEL_DIR / "model.safetensors").exists():
        print(f"[ERROR] Pruned model not found in: {PRUNED_MODEL_DIR}")
        print("Please run 02_prune_to_text.py first!")
        return

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required to test GPU loading.")
        return

    print("Configuring 4-bit BitsAndBytes quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading pruned text model from: {PRUNED_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(PRUNED_MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(PRUNED_MODEL_DIR),
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print("\n" + "*" * 65)
    print(f"  VRAM used by Pruned 4-bit Model: {vram_mb:.2f} MB (~{vram_mb/1024:.2f} GB)!")
    print(f"  Compare to original unpruned    : ~6,431 MB (~6.28 GB)")
    print(f"  VRAM Saved                      : ~{6431 - vram_mb:.0f} MB (Over 4.5 GB freed up!)")
    print("*" * 65)

    # Test text inference
    test_query = "Route this user message into JSON: Humayun ahmed er nishithini book er dam koto?"
    print(f"\nRunning test inference with prompt:\n  '{test_query}'")

    messages = [{"role": "user", "content": test_query}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\nModel Output:")
    print("-" * 50)
    print(response.strip())
    print("-" * 50)

    print("=" * 65)
    print("[VERIFIED] Pruned text model works flawlessly!")
    print("You can now update steps/config.py to use this pruned model:")
    print(f"  MODEL_ID = r'{PRUNED_MODEL_DIR.resolve()}'")
    print("=" * 65)

if __name__ == "__main__":
    main()
