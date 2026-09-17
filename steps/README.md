# Step-by-Step QLoRA Fine-Tuning Guide

This folder breaks down the fine-tuning pipeline into sequential, self-contained Python scripts. You can run them one by one to inspect, verify, and understand each stage.

---

### Execution Order

```powershell
cd "d:\Project\Fine Tune\steps"
```

1. **`python 01_check_environment.py`**
   - Verifies PyTorch version, CUDA GPU device name, VRAM, bfloat16 support, and dependencies.

2. **`python 02_prepare_dataset.py`**
   - Inspects `dataset_train.jsonl` and `dataset_val.jsonl`.
   - Formats queries using Gemma's ChatML template and displays token length statistics.

3. **`python 03_load_base_model.py`**
   - Loads the base model in 4-bit NormalFloat (NF4).
   - Measures base model VRAM usage on your RTX 5050.
   - Runs a zero-shot test to see how the model behaves **before** fine-tuning.

4. **`python 04_setup_lora.py`**
   - Attaches the LoRA adapter (rank 16, alpha 32) to attention and MLP projection layers.
   - Shows trainable parameters (~0.5%) vs frozen base model parameters.

5. **`python 05_train.py`**
   - Runs the actual QLoRA training using `SFTTrainer`.
   - Logs loss every 10 steps, evaluates on validation data, and saves checkpoints.
   - Saves final adapter weights to `../gemma-router-qlora/final_adapter`.

6. **`python 06_test_adapter.py`**
   - Loads base model + trained adapter.
   - Evaluates test queries across Bangla, Banglish, and English to verify pure JSON output.

7. **`python 07_merge_and_export.py`**
   - Merges the adapter weights back into the base model weights on CPU.
   - Outputs the merged standalone model to `../gemma-router-merged`.
   - Displays exact instructions for GGUF quantization and Ollama registration.

---

### Configuration
All file paths, model IDs, and hyperparameters are centrally managed in [`config.py`](config.py).
