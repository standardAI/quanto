import os

from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
cpu_lib = load(name="quanto_cpu", sources=[f"{module_path}/unpack.cpp"], extra_cflags=["-O3"])
