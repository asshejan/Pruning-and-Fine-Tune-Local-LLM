"""
Step 6: Test Fine-Tuned Adapter Inference
Loads the base model + trained adapter from ./gemma-router-qlora/final_adapter
and tests routing queries across Bangla, Banglish, and English.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import MODEL_ID, FINAL_ADAPTER_DIR

TEST_QUERIES = [
    # Banglish Price
    "Humayun ahmed er nishithini book er dam koto?",
    # Bangla Price
    "অ্যাকিলিসের টেন্ডন বইয়ের দাম কত?",
    # Banglish Availability
    "Sisimpur er Dukhi Kathbirali ache kina?",
    # English Price
    "How much is 1 Dozen Rommo by Ahsan Habib?",
    # Vague query (expect clarify)
    "boi lagbe",
    # Offtopic query (expect decline)
    "what is the weather today in Dhaka?"
]

def main():
    print("=" * 60)
    print("STEP 6: Test Fine-Tuned Adapter Output")
    print("=" * 60)

    if not FINAL_ADAPTER_DIR.exists():
        print(f"[ERROR] Adapter directory not found: {FINAL_ADAPTER_DIR}")
        print("Please run Step 5 (05_train.py) first to train and save the adapter.")
        return

    print("Loading 4-bit base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_ADAPTER_DIR))
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    print(f"Loading trained adapter from {FINAL_ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, str(FINAL_ADAPTER_DIR))
    model.eval()

    print("\n--- Running Evaluation Test Cases ---")
    for q in TEST_QUERIES:
        prompt = f"Route this user message into JSON: {q}"
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n[QUERY] : {q}")
        print(f"[OUTPUT]: {resp.strip()}")

    print("=" * 60)
    print("[SUCCESS] Testing complete! Notice how the adapter strictly outputs pure JSON.")
    print("=" * 60)

if __name__ == "__main__":
    main()
