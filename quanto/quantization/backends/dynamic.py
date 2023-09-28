import torch

from ...transform import TransformBackend
from ..nn import DynamicQLinear


class DynamicQuantizationBackend(TransformBackend):
    """
    Apply quantization transformations to a graph when calling `torch.compile`.

    The transformations are applied to the graph before calling the underlying optimization
    backend.

    Args:
        backend (str): The name of the backend to use for optimization.
    """

    def __init__(self, backend: str):
        def quantize_linear(linear: torch.nn.Linear):
            return DynamicQLinear.from_module(linear)

        super().__init__(backend, replacers={torch.nn.Linear: quantize_linear})
