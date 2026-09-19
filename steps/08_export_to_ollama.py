"""
Step 8: Export Merged Model to GGUF and Register in Ollama
Uses llama.cpp's convert_hf_to_gguf.py to export a Q8_0 GGUF model (~4.9GB)
and registers it with Ollama under gemma4-router:latest.
"""
import subprocess
import sys
from pathlib import Path
from config import WORKSPACE_DIR, MERGED_OUTPUT_DIR

GGUF_OUTFILE = WORKSPACE_DIR / "gemma-router-q8_0.gguf"
MODELFILE_PATH = WORKSPACE_DIR / "Modelfile"
LLAMA_CPP_CONVERT = WORKSPACE_DIR / "llama.cpp" / "convert_hf_to_gguf.py"
OLLAMA_MODEL_NAME = "gemma4-router:latest"

def main():
    print("=" * 60)
    print("STEP 8: Export to GGUF & Register in Ollama")
    print("=" * 60)

    if not MERGED_OUTPUT_DIR.exists():
        print(f"[ERROR] Merged model directory not found: {MERGED_OUTPUT_DIR}")
        print("Please run Step 7 (07_merge_and_export.py) first.")
        return

    # 1. Check llama.cpp conversion script
    if not LLAMA_CPP_CONVERT.exists():
        print(f"llama.cpp conversion script not found at {LLAMA_CPP_CONVERT}.")
        print("Cloning shallow llama.cpp repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(WORKSPACE_DIR / "llama.cpp")],
            check=True
        )

    # 2. Convert to GGUF
    print(f"\n1. Converting {MERGED_OUTPUT_DIR} to Q8_0 GGUF...")
    cmd_convert = [
        sys.executable,
        str(LLAMA_CPP_CONVERT),
        str(MERGED_OUTPUT_DIR),
        "--outtype", "q8_0",
        "--outfile", str(GGUF_OUTFILE)
    ]
    subprocess.run(cmd_convert, check=True)
    print(f"[OK] GGUF saved to {GGUF_OUTFILE}")

    # 3. Create Modelfile
    print(f"\n2. Creating Ollama Modelfile at {MODELFILE_PATH}...")
    modelfile_content = (
        f"FROM ./{GGUF_OUTFILE.name}\n\n"
        'PARAMETER stop "<turn|>"\n'
        'PARAMETER stop "<|turn>"\n'
        'PARAMETER temperature 0.1\n'
    )
    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    # 4. Register with Ollama
    print(f"\n3. Registering model with Ollama as '{OLLAMA_MODEL_NAME}'...")
    subprocess.run(
        ["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(MODELFILE_PATH)],
        check=True
    )

    print("=" * 60)
    print(f"[SUCCESS] Model '{OLLAMA_MODEL_NAME}' is now ready in Ollama!")
    print(f"Test it with: ollama run {OLLAMA_MODEL_NAME} 'Your query here'")
    print("=" * 60)

if __name__ == "__main__":
    main()
