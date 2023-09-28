import torch
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from quanto.quantization.backends import DynamicQuantizationBackend


def test_quantize_attention():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Instantiate a Llama attention module
    config = LlamaConfig()
    att = LlamaAttention(config).to(device)

    # Get reference outputs
    hidden_states = torch.randn((1, 128, config.hidden_size)).to(device)
    with torch.no_grad():
        out = att(hidden_states)

    # Instantiate and configure a quantization backend
    qb = DynamicQuantizationBackend("inductor")

    # Compile the module with the backend to apply transformations before compilation
    catt = torch.compile(att, backend=qb)

    # Compare outputs
    with torch.no_grad():
        cout = catt(hidden_states)

    # W/O any transformations, results are identical
    assert torch.allclose(out[0], cout[0], atol=1e-7)
