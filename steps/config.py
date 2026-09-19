"""
Shared configuration for the step-by-step QLoRA fine-tuning pipeline.
"""
from pathlib import Path

# Directories & files
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MODEL_ID = str(WORKSPACE_DIR / "gemma-4-E2B-text")
DATA_TRAIN = WORKSPACE_DIR / "dataset_train.jsonl"
DATA_VAL = WORKSPACE_DIR / "dataset_val.jsonl"

ADAPTER_OUTPUT_DIR = WORKSPACE_DIR / "gemma-router-qlora"
FINAL_ADAPTER_DIR = ADAPTER_OUTPUT_DIR / "final_adapter"
MERGED_OUTPUT_DIR = WORKSPACE_DIR / "gemma-router-merged"

# LoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training hyperparameters (optimized for 8GB VRAM RTX 5050)
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512
