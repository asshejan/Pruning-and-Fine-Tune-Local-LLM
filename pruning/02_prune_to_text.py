"""
Pruning Step 2: Extract & Save Text-Only Model
Discards all Vision & Audio weights and exports a native, first-class
Gemma4ForCausalLM text model to ../gemma-4-E2B-text.
"""
import os
import glob
from pathlib import Path
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

SOURCE_MODEL_ID = "google/gemma-4-E2B-it"
TARGET_DIR = Path(__file__).resolve().parent.parent / "gemma-4-E2B-text"

def main():
    print("=" * 65)
    print("STEP 2: Pruning to Pure Text Model (Gemma4ForCausalLM)")
    print("=" * 65)

    # 1. Find cached safetensors file
    cache_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*/*.safetensors"
    )
    files = glob.glob(cache_pattern)
    if not files:
        print("[ERROR] Cached model.safetensors not found.")
        return

    source_file = files[0]
    print(f"Loading weights from:\n  {source_file}...")
    weights = load_file(source_file, device="cpu")
    print(f"Loaded {len(weights)} total weight tensors.")

    # 2. Extract and remap text language model tensors
    # In multimodal Gemma4, text tensors are under: model.language_model.*
    # In native Gemma4ForCausalLM, text tensors are under: model.*
    print("\nFiltering and remapping text language model tensors...")
    text_weights = {}
    skipped_vision = 0
    skipped_audio = 0

    for k, tensor in weights.items():
        if "vision_tower" in k:
            skipped_vision += 1
            continue
        if "audio_tower" in k:
            skipped_audio += 1
            continue

        # Remap model.language_model. -> model.
        if k.startswith("model.language_model."):
            new_key = "model." + k[len("model.language_model."):]
            text_weights[new_key] = tensor
        else:
            text_weights[k] = tensor

    print(f"  [+] Kept text tensors : {len(text_weights)}")
    print(f"  [-] Dropped vision    : {skipped_vision}")
    print(f"  [-] Dropped audio     : {skipped_audio}")

    # 3. Create destination folder
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    out_safetensors = TARGET_DIR / "model.safetensors"
    print(f"\nSaving pruned weights to:\n  {out_safetensors}...")
    save_file(text_weights, str(out_safetensors))

    # 4. Save native text config
    print("Configuring standalone Gemma4TextConfig...")
    full_config = AutoConfig.from_pretrained(SOURCE_MODEL_ID)
    text_config = full_config.text_config
    text_config.architectures = ["Gemma4ForCausalLM"]
    text_config.save_pretrained(str(TARGET_DIR))

    # 5. Save Tokenizer
    print("Saving tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_ID)
    tokenizer.save_pretrained(str(TARGET_DIR))

    # 6. Report new disk size
    new_size_gb = os.path.getsize(out_safetensors) / (1024 ** 3)
    print("=" * 65)
    print(f"[SUCCESS] Pure text model created successfully!")
    print(f"Location      : {TARGET_DIR.resolve()}")
    print(f"New Model Size: {new_size_gb:.2f} GB (Down from 10.2 GB!)")
    print("=" * 65)

if __name__ == "__main__":
    main()
