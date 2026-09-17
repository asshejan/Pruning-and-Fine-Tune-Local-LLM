"""
Step 2: Inspect & Tokenize Dataset
Loads the generated JSONL dataset, applies the chat template,
and checks token lengths and prompt formatting.
"""
from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_ID, DATA_TRAIN, DATA_VAL, MAX_SEQ_LENGTH

def main():
    print("=" * 60)
    print("STEP 2: Dataset Preparation & Tokenization Check")
    print("=" * 60)

    if not DATA_TRAIN.exists() or not DATA_VAL.exists():
        print(f"[ERROR] Dataset files not found in workspace!")
        print(f"Please ensure {DATA_TRAIN} and {DATA_VAL} exist.")
        return

    print(f"Loading training data from  : {DATA_TRAIN}")
    print(f"Loading validation data from: {DATA_VAL}")

    dataset = load_dataset("json", data_files={
        "train": str(DATA_TRAIN),
        "validation": str(DATA_VAL)
    })

    print(f"\nTotal train samples: {len(dataset['train'])}")
    print(f"Total val samples  : {len(dataset['validation'])}")

    print(f"\nLoading tokenizer for: {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        if "gated repo" in str(e).lower() or "401" in str(e) or "restricted" in str(e).lower():
            print("\n" + "!" * 60)
            print("[AUTHENTICATION REQUIRED]")
            print(f"'{MODEL_ID}' is a gated model on Hugging Face.")
            print("1. Visit: https://huggingface.co/google/gemma-2-2b-it and accept Google's license terms.")
            print("2. Login in your terminal by running:")
            print("     hf auth login")
            print("   or set your Hugging Face token in PowerShell:")
            print("     $env:HF_TOKEN = 'your_hf_token'")
            print("!" * 60 + "\n")
            return
        raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n--- Example 1 Raw Message Format ---")
    sample_conv = dataset["train"][0]["messages"]
    print(sample_conv)

    print("\n--- Formatted with Model's Chat Template ---")
    formatted_text = tokenizer.apply_chat_template(sample_conv, tokenize=False)
    print(formatted_text)

    # Token length statistics
    print("\nCalculating token length distribution...")
    token_lengths = []
    for row in dataset["train"]:
        tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True)
        token_lengths.append(len(tokens))

    min_len = min(token_lengths)
    max_len = max(token_lengths)
    avg_len = sum(token_lengths) / len(token_lengths)

    print(f"  Min tokens : {min_len}")
    print(f"  Max tokens : {max_len}")
    print(f"  Avg tokens : {avg_len:.1f}")
    print(f"  Max sequence limit set to: {MAX_SEQ_LENGTH}")

    if max_len > MAX_SEQ_LENGTH:
        print(f"  [WARNING] Some samples ({max_len} tokens) exceed MAX_SEQ_LENGTH ({MAX_SEQ_LENGTH}) and will be truncated.")
    else:
        print(f"  [OK] All samples fit comfortably within {MAX_SEQ_LENGTH} tokens without truncation!")

    print("=" * 60)
    print("[SUCCESS] Dataset is validated and ready for training!")
    print("=" * 60)

if __name__ == "__main__":
    main()
