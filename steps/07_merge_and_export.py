"""
Step 7: Merge Adapter & Prepare for Ollama Export
Merges the LoRA adapter back into base model weights on CPU
and outputs instructions for llama.cpp GGUF conversion & Ollama Modelfile.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import MODEL_ID, FINAL_ADAPTER_DIR, MERGED_OUTPUT_DIR

def main():
    print("=" * 60)
    print("STEP 7: Merge LoRA Adapter into Base Model")
    print("=" * 60)

    if not FINAL_ADAPTER_DIR.exists():
        print(f"[ERROR] Adapter directory not found: {FINAL_ADAPTER_DIR}")
        print("Please run Step 5 (05_train.py) first.")
        return

    print(f"1. Loading base model {MODEL_ID} in float16 onto CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu"  # CPU merge avoids GPU VRAM limits
    )

    print(f"2. Attaching adapter from {FINAL_ADAPTER_DIR}...")
    lora_model = PeftModel.from_pretrained(base_model, str(FINAL_ADAPTER_DIR))

    print("3. Merging adapter weights into base weights (merge_and_unload)...")
    merged_model = lora_model.merge_and_unload()

    print(f"4. Saving merged standalone model to:\n   {MERGED_OUTPUT_DIR}...")
    MERGED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(MERGED_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MERGED_OUTPUT_DIR))

    print("=" * 60)
    print("[SUCCESS] Merged model successfully saved!")
    print("=" * 60)
    print("\n--- NEXT STEPS FOR OLLAMA DEPLOYMENT ---")
    print("1. Convert to GGUF using llama.cpp:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {MERGED_OUTPUT_DIR} --outtype f16 --outfile gemma-router-f16.gguf")
    print("\n2. Quantize to Q4_K_M:")
    print("   ./llama.cpp/llama-quantize gemma-router-f16.gguf gemma-router-q4_k_m.gguf Q4_K_M")
    print("\n3. Create an Ollama Modelfile:")
    print("   FROM ./gemma-router-q4_k_m.gguf")
    print('   TEMPLATE "{{ .Prompt }}"')
    print('   PARAMETER temperature 0.1')
    print('   PARAMETER stop "<turn|>"')
    print("\n4. Register in Ollama:")
    print("   ollama create gemma4-e2b-router:latest -f Modelfile")
    print("=" * 60)

if __name__ == "__main__":
    main()
