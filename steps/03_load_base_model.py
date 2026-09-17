"""
Step 3: Test 4-bit Base Model Loading & Baseline Inference
Loads the base model quantized in 4-bit NormalFloat (NF4)
and runs an untuned query to observe base model behavior.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODEL_ID

def main():
    print("=" * 60)
    print("STEP 3: 4-bit Base Model Loading & Baseline Inference")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required to test 4-bit bitsandbytes loading.")
        return

    print("Configuring 4-bit BitsAndBytes quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading tokenizer & 4-bit model from: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    vram_used = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"\n[OK] Model loaded successfully into GPU!")
    print(f"VRAM used by 4-bit base model: {vram_used:.2f} MB (~{vram_used/1024:.2f} GB)")

    # Run a test query with the UNTUNED base model
    test_query = "Route this user message into JSON: Humayun ahmed er nishithini book er dam koto?"
    print("\n--- Testing Baseline Zero-Shot Output (Before Fine-Tuning) ---")
    print(f"Prompt: {test_query}")

    messages = [{"role": "user", "content": test_query}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\nModel Output (Untuned Base):")
    print("-" * 40)
    print(response.strip())
    print("-" * 40)
    print("\nNotice: The untuned base model may output conversational preamble or wrong formatting.")
    print("Fine-tuning in Step 5 will train it to produce exact, pure JSON schema.")
    print("=" * 60)

if __name__ == "__main__":
    main()
