import os
from pathlib import Path

import clip
import torch
from onnxruntime.quantization import quantize_dynamic

if __name__ == "__main__":
    clean_name = "clip-vit-base-patch32"
    onnx_name = os.path.join("models", f"{clean_name}.onnx")

    model, preprocess = clip.load("ViT-B/32", device="cpu")
    dummy_input = clip.tokenize(["dummy input text"])

    # Hack
    model.forward = model.encode_text

    torch.onnx.export(
        model,
        dummy_input,
        onnx_name,
        input_names=["input_ids"],
        output_names=["output"],
        export_params=True,
        dynamic_axes={"input_ids": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    quantize_dynamic(Path(onnx_name), Path(onnx_name.replace(".onnx", "-q8.onnx")))
    os.remove(f"{clean_name}.onnx")
