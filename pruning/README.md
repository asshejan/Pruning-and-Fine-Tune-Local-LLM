# Gemma 4 Multimodal Pruning Pipeline

This folder provides a 3-step workflow to strip away the **Vision Tower** and **Audio Tower** from `google/gemma-4-E2B-it`, leaving a clean, lightweight **Text-Only Gemma 4 CausalLM** for router fine-tuning.

---

### Why Prune?

* **Original Multimodal Model**: **10.2 GB** on disk, **~6.3 GB VRAM** in 4-bit.
* **Pruned Text Model**: **~3.2 GB** on disk, **~1.5 GB VRAM** in 4-bit.
* **VRAM Saved**: **~4.7 GB of GPU VRAM freed up** on your RTX 5050 for training!

---

### Step-by-Step Execution

```powershell
cd "d:\Project\Fine Tune\pruning"
```

1. **`python 01_inspect_towers.py`**
   - Inspects the cached model weights and prints the parameter breakdown across Vision, Audio, and Text.

2. **`python 02_prune_to_text.py`**
   - Discards all vision and audio weights.
   - Saves the clean text language model to `../gemma-4-E2B-text`.

3. **`python 03_verify_pruned_model.py`**
   - Loads the pruned model in 4-bit on your GPU.
   - Measures the new VRAM footprint (~1.5 GB) and tests text generation.

---

### After Pruning
Update `steps/config.py`:
```python
MODEL_ID = r"d:\Project\Fine Tune\gemma-4-E2B-text"
```
Then proceed with fine-tuning in `steps/`!
