"""
Step 1: Verify Hardware, CUDA, and Training Libraries
Run this script first to confirm GPU capability and library readiness.
"""
import sys
import torch

def main():
    print("=" * 60)
    print("STEP 1: Hardware & Environment Verification")
    print("=" * 60)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"PyTorch version   : {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available    : {cuda_available}")

    if not cuda_available:
        print("\n[WARNING] CUDA is NOT detected by PyTorch in this environment!")
        print("To enable GPU training on your RTX 5050, run:")
        print("  pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    bf16_ok = torch.cuda.is_bf16_supported()

    print(f"GPU Device        : {gpu_name}")
    print(f"Total VRAM        : {vram_gb:.2f} GB")
    print(f"bfloat16 supported: {bf16_ok}")

    # Check key QLoRA dependencies
    print("\nChecking training packages...")
    packages = [
        ("transformers", "Transformers"),
        ("peft", "PEFT (LoRA)"),
        ("bitsandbytes", "BitsAndBytes (4-bit quantization)"),
        ("trl", "TRL (SFTTrainer)"),
        ("datasets", "Hugging Face Datasets"),
        ("accelerate", "Accelerate"),
    ]

    all_ok = True
    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {label:<35} : v{ver}")
        except ImportError as e:
            print(f"  [MISSING] {label:<30} : {e}")
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("[SUCCESS] Environment is fully configured for QLoRA training!")
    else:
        print("[ACTION NEEDED] Please install missing dependencies before proceeding.")
    print("=" * 60)

if __name__ == "__main__":
    main()
