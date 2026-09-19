"""
Direct Interactive Chat with Pruned Gemma 4 E2B (Safetensors / 4-bit).
Run this script to chat directly with the model in real time before fine-tuning.
"""
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

MODEL_PATH = Path(__file__).resolve().parent / "gemma-4-E2B-text"

def main():
    print("=" * 65)
    print("Gemma 4 E2B (Pruned Text Model) - Direct Interactive Chat")
    print("=" * 65)

    if not MODEL_PATH.exists() or not (MODEL_PATH / "model.safetensors").exists():
        print(f"[ERROR] Pruned model directory not found: {MODEL_PATH}")
        print("Please ensure pruning/02_prune_to_text.py has finished.")
        return

    print("1. Configuring 4-bit BitsAndBytes quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print(f"2. Loading model from {MODEL_PATH.resolve()} into GPU...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"\n[OK] Model successfully loaded into GPU!")
    print(f"VRAM Allocated: {vram_mb:.1f} MB (~{vram_mb/1024:.2f} GB)\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("-" * 65)
    print("Ready! Type your question in Bangla, Banglish, or English.")
    print("Type 'exit' or 'quit' to close.")
    print("-" * 65)

    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        # Test prompt
        conversation_history.append({"role": "user", "content": user_input})
        prompt_text = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        print("Gemma: ", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id
            )

        # Save model reply to conversation
        reply_tokens = outputs[0][inputs.input_ids.shape[1]:]
        model_reply = tokenizer.decode(reply_tokens, skip_special_tokens=True)
        conversation_history.append({"role": "model", "content": model_reply})

if __name__ == "__main__":
    main()
