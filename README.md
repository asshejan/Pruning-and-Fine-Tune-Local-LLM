# BoiBazar Router: Gemma 4 QLoRA Fine-Tuning & Pruning Pipeline

This repository contains the end-to-end pipeline for **pruning**, **fine-tuning**, and **exporting** Google's **`gemma-4-E2B-it`** model to serve as the ultra-fast, structured query router for the **BoiBazar** book-buying assistant.

---

## 📖 Overview

The BoiBazar customer service flow receives queries in **Bangla**, **Banglish** (Bengali written in Latin letters), and **English**. The router model must:
1. Classify the user intent (`price`, `availability`, `author`, `vague`, `offtopic`).
2. Route into decision branches (`book_lookup`, `clarify`, `decline`).
3. Extract catalog entities (`title`, `author`, `publisher`).
4. Output **strictly valid JSON** without conversational preamble ("chattering").

By pruning Gemma 4's multimodal towers and fine-tuning with **QLoRA (4-bit)**, we achieve high accuracy, sub-second latency, and a lightweight VRAM footprint (~2 GB) suitable for local deployment on an **8 GB GPU (e.g., NVIDIA GeForce RTX 5050)**.

---

## 🏗️ Architecture & Key Innovations

### 1. Multimodal Tower Pruning
`google/gemma-4-E2B-it` is natively a multimodal model containing **2,011 tensors** (~10.2 GB raw):
* 👁️ **Vision Tower**: 659 tensors
* 🎧 **Audio Tower**: 752 tensors
* 📖 **Text Language Model**: 600 tensors

Since BoiBazar query routing is purely text-based, the vision and audio encoders are 100% dead weight. Our pruning step strips **1,411 audio and vision tensors**, remapping the weights into a native, standalone **`Gemma4ForCausalLM`** model that loads in only **~2.2 GB VRAM** in 4-bit.

### 2. QLoRA (4-bit Quantized Low-Rank Adaptation)
* **Base Weights**: Frozen in 4-bit NormalFloat (`nf4`) with double quantization.
* **Trainable Parameters**: Lightweight LoRA adapters ($r=16, \alpha=32$) attached to attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and feedforward layers (`gate_proj`, `up_proj`, `down_proj`).
* **VRAM Footprint**: Under 3.5 GB during training on an 8 GB RTX 5050 using `paged_adamw_8bit`.

---

## 📂 Repository Structure

```
.
├── .gitignore                     # Excludes datasets, virtualenvs, and raw weight files
├── README.md                      # Project documentation (this file)
├── extract_dataset.py             # Extracts real catalog books from OpenSearch & synthesizes train/val splits
│
├── pruning/                       # Module 1: Multimodal Pruning Pipeline
│   ├── 01_inspect_towers.py       # Step 1: Breakdown of Vision, Audio, and Text parameters
│   ├── 02_prune_to_text.py        # Step 2: Strips 1,411 audio/vision tensors & exports text model
│   ├── 03_verify_pruned_model.py  # Step 3: Tests the pruned model in 4-bit & verifies VRAM savings
│   └── README.md                  # Detailed pruning documentation
│
└── steps/                         # Module 2: Step-by-Step QLoRA Fine-Tuning Pipeline
    ├── config.py                  # Central configuration (paths, hyperparameters, model IDs)
    ├── 01_check_environment.py    # Step 1: Verifies CUDA, PyTorch, and training dependencies
    ├── 02_prepare_dataset.py      # Step 2: Inspects tokenization & validates ChatML template
    ├── 03_load_base_model.py      # Step 3: Tests 4-bit model loading & baseline zero-shot output
    ├── 04_setup_lora.py           # Step 4: Attaches LoRA adapter & inspects trainable parameter count
    ├── 05_train.py                # Step 5: Runs QLoRA fine-tuning with SFTTrainer & saves adapter
    ├── 06_test_adapter.py         # Step 6: Tests fine-tuned router inference on test queries
    ├── 07_merge_and_export.py     # Step 7: Merges adapter with base weights for Ollama / GGUF export
    └── README.md                  # Step-by-step execution guide
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Ensure you have a Python 3.11 environment with CUDA-enabled PyTorch:
```powershell
pip install -r requirements.txt # or install: torch transformers peft trl datasets bitsandbytes accelerate
```

Verify GPU and CUDA readiness:
```powershell
python steps/01_check_environment.py
```

---

### 2. Generate Dataset from OpenSearch

Extract real catalog books and user queries from your local OpenSearch instance:
```powershell
python extract_dataset.py
```
* Generates `dataset_train.jsonl` (1,458 samples) and `dataset_val.jsonl` (162 samples) formatted in Gemma turn template.

---

### 3. Prune Multimodal Towers (Vision & Audio)

Strip out unnecessary vision and audio weights to reduce VRAM from 6.3 GB to ~2.2 GB:
```powershell
cd pruning
python 01_inspect_towers.py      # Inspect the 3 component towers
python 02_prune_to_text.py       # Extract text weights to ../gemma-4-E2B-text
python 03_verify_pruned_model.py # Verify 4-bit GPU loading and generation
cd ..
```

---

### 4. Run QLoRA Fine-Tuning (Step-by-Step)

Navigate to `steps/` and run the modular scripts sequentially:

```powershell
cd steps

# Inspect dataset formatting and token lengths
python 02_prepare_dataset.py

# Load base model in 4-bit and inspect untuned baseline output
python 03_load_base_model.py

# Attach LoRA adapter (inspects trainable parameters: ~0.5%)
python 04_setup_lora.py

# Run QLoRA fine-tuning (takes ~3-5 mins on RTX 5050)
python 05_train.py

# Test fine-tuned adapter across Bangla, Banglish, and English
python 06_test_adapter.py

# Merge adapter weights into standalone model for GGUF / Ollama export
python 07_merge_and_export.py
```

---

### 5. Export to Ollama

Once Step 7 merges the model into `gemma-router-merged/`:

1. **Convert to GGUF**:
   ```powershell
   python llama.cpp/convert_hf_to_gguf.py ./gemma-router-merged --outtype f16 --outfile gemma-router-f16.gguf
   ```
2. **Quantize to Q4_K_M**:
   ```powershell
   ./llama.cpp/llama-quantize gemma-router-f16.gguf gemma-router-q4_k_m.gguf Q4_K_M
   ```
3. **Register in Ollama**:
   Create a `Modelfile`:
   ```dockerfile
   FROM ./gemma-router-q4_k_m.gguf
   TEMPLATE "{{ .Prompt }}"
   PARAMETER temperature 0.1
   PARAMETER stop "<turn|>"
   ```
   And run:
   ```powershell
   ollama create gemma4-e2b-router:latest -f Modelfile
   ```

---

## ⚙️ Hyperparameters (`steps/config.py`)

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| **LoRA Rank ($r$)** | `16` | Optimal capacity for closed-domain routing & entity extraction. |
| **LoRA Alpha ($\alpha$)** | `32` | Standard scaling ratio ($\alpha = 2 \times r$). |
| **LoRA Dropout** | `0.05` | Prevents title memorization / overfitting. |
| **Target Modules** | Attention + MLP | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` |
| **Batch Size** | `2` | Guarantees zero OOM on 8 GB VRAM. |
| **Gradient Accumulation** | `4` | Effective batch size = $8$. |
| **Learning Rate** | `2e-4` | Stable AdamW convergence for PEFT. |
| **Max Sequence Length** | `512` | Short context window saves massive memory. |
| **Optimizer** | `paged_adamw_8bit` | Offloads optimizer states during memory spikes. |

---

## 🔒 Data Privacy & Leakage Prevention

* `dataset_train.jsonl` and `dataset_val.jsonl` are strictly generated using synthetic template permutations on live catalog titles and are ignored by `.gitignore`.
* Production evaluation suites (`tests/eval_1000.jsonl`) remain strictly held-out to maintain zero benchmark test-set contamination.
