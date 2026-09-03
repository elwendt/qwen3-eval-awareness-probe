from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(
    "Qwen/Qwen3-8B",
    device="cuda"
)

print("Qwen3-8B loaded successfully")
