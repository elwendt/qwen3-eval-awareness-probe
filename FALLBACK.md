# Fallback: if TransformerLens doesn't load Qwen3-8B

`src/model_loader.py` tries `HookedTransformer.from_pretrained("Qwen/Qwen3-8B")`
first. If that fails after checking you're on the latest TransformerLens
(PyPI or `main` branch -- see the error message `model_loader.py` raises),
the fallback is raw `transformers` + manual forward hooks. More setup work,
but the underlying idea is identical to what you already did with GPT-2's
hook points: you're recreating the one specific piece of what
TransformerLens automates for you.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

captured = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # output[0] is the residual stream after this decoder layer for most
        # HF causal-LM implementations -- CONFIRM this against Qwen3's actual
        # modeling code before trusting it. Don't assume the index matches
        # other model families just because it usually does.
        captured[layer_idx] = output[0].detach()
    return hook

for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(make_hook(i))

inputs = tokenizer("test prompt", return_tensors="pt").to("cuda")
with torch.no_grad():
    model(**inputs)

# captured[i] now holds the residual stream at layer i, shape (1, seq_len, d_model)
```

If you end up here, verify `output[0]` against Qwen3's actual modeling
source (`transformers.models.qwen3.modeling_qwen3`) rather than trusting
this snippet's assumption -- exactly the "verify against source, don't
trust a remembered API shape" discipline that's caught bugs in your work
before.
