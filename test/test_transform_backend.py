import pytest
import torch
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from quanto.transform import TransformBackend


class EmptyBackend(TransformBackend):
    pass


class CopyLinear(torch.nn.Linear):
    @classmethod
    def from_module(cls, module):
        new_module = cls(module.in_features, module.out_features, module.bias is not None)
        new_module.weight.copy_(module.weight)
        if module.bias is not None:
            new_module.bias.copy_(module.bias)
        return new_module.to(module.weight.device)


class CopyLinearBackend(TransformBackend):
    def __init__(self, backend: str):
        def copy_linear(linear: torch.nn.Linear):
            return CopyLinear.from_module(linear)

        super().__init__(backend, replacers={torch.nn.Linear: copy_linear})


@pytest.mark.parametrize("backend_class", [EmptyBackend, CopyLinearBackend])
def test_transform_attention(backend_class):
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

    # Instantiate the transform backend
    backend = backend_class("inductor")

    # Reset dynamo before using a new backend
    torch._dynamo.reset()

    # Compile the module with the backend to apply transformations before compilation
    catt = torch.compile(att, backend=backend)

    # Compare outputs
    with torch.no_grad():
        cout = catt(hidden_states)

    # W/O any transformations, results are identical
    assert torch.allclose(out[0], cout[0], atol=1e-7)
