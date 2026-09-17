"""
Pruning Step 1: Inspect Multimodal Towers
Analyzes the cached google/gemma-4-E2B-it weights and shows
the exact tensor count, parameter count, and size breakdown
for Vision, Audio, and Text.
"""
import os
import glob
from safetensors import safe_open

def main():
    print("=" * 65)
    print("STEP 1: Multimodal Tower Parameter & Tensor Inspection")
    print("=" * 65)

    cache_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*/*.safetensors"
    )
    files = glob.glob(cache_pattern)
    if not files:
        print("[ERROR] Cached model.safetensors not found.")
        print("Please make sure Step 3 was run so the model is cached.")
        return

    weight_file = files[0]
    file_size_gb = os.path.getsize(weight_file) / (1024 ** 3)
    print(f"Inspecting weight file:\n  {weight_file}")
    print(f"Total file size on disk: {file_size_gb:.2f} GB\n")

    vision_params, audio_params, text_params = 0, 0, 0
    vision_tensors, audio_tensors, text_tensors = 0, 0, 0

    with safe_open(weight_file, framework="pt", device="cpu") as f:
        for k in f.keys():
            shape = f.get_slice(k).get_shape()
            num_params = 1
            for dim in shape:
                num_params *= dim

            if "vision" in k:
                vision_tensors += 1
                vision_params += num_params
            elif "audio" in k:
                audio_tensors += 1
                audio_params += num_params
            else:
                text_tensors += 1
                text_params += num_params

    total_params = vision_params + audio_params + text_params
    total_tensors = vision_tensors + audio_tensors + text_tensors

    print("-" * 65)
    print(f"{'Component':<20} | {'Tensors':<10} | {'Parameters':<14} | {'Share (%)':<10}")
    print("-" * 65)
    print(f"{'Vision Tower':<20} | {vision_tensors:<10} | {vision_params/1e6:>8.2f} M   | {vision_params/total_params*100:>8.1f} %")
    print(f"{'Audio Tower':<20} | {audio_tensors:<10} | {audio_params/1e6:>8.2f} M   | {audio_params/total_params*100:>8.1f} %")
    print(f"{'Text Language Model':<20} | {text_tensors:<10} | {text_params/1e6:>8.2f} M   | {text_params/total_params*100:>8.1f} %")
    print("-" * 65)
    print(f"{'TOTAL':<20} | {total_tensors:<10} | {total_params/1e9:>8.2f} B   | {'100.0 %':>8}")
    print("-" * 65)

    removable_params = vision_params + audio_params
    print(f"\n[INSIGHT] Pruning Vision + Audio removes {removable_params/1e6:.1f} Million parameters ({removable_params/total_params*100:.1f}% of the model)!")
    print(f"The text model alone is only ~{text_params/1e9:.2f} Billion parameters.")
    print("=" * 65)

if __name__ == "__main__":
    main()
