# Download models in advance
from transformers import AutoModelForCausalLM
import torch
models_to_cache = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf"
]

for model_name in models_to_cache:
    print(f"Caching {model_name}...")
    # This downloads and caches the model
    AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    print(f"Cached {model_name}")
