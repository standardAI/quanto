from abc import ABC
from typing import Callable, Dict, Type, Union

import torch
from torch._dynamo.backends.registry import lookup_backend
from torch.fx.experimental.optimization import replace_node_module


Target = Union[Type[torch.nn.Module], Callable]
Replacer = Callable


class TransformBackend(ABC):
    """
    This is the base class to apply modifications on a graph when calling `torch.compile`.

    It is composable with optimization backends: it modifies the FX graph produced by Torch Dynamo
    before passing it to the actual graph optimization backend.

    The actual graph transformations must be implemented in subclasses.

    Args:
        backend (str): The name of the backend to use for optimization.
    """

    def __init__(self, backend, replacers: Dict[Target, Replacer] = {}):
        self.backend = lookup_backend(backend)
        self.replacers = replacers

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        return self.backend(self.transform(gm), example_inputs)

    def transform(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        modules = dict(gm.named_modules())
        for n in gm.graph.nodes:
            if n.op == "call_module":
                module = modules[n.target]
                module_type = type(module)
                if module_type in self.replacers:
                    new_module = self.replacers[module_type](module)
                    replace_node_module(n, modules, new_module)
        return gm
