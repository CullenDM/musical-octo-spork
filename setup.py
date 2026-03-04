"""
Setup script for musical-octo-spork.

Builds the optional CUDA extension (`mospork_kernels`) when CUDA is available.
The model code works without the extension but will run faster with it.
"""

import os
from setuptools import setup, find_packages

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    import torch

    cuda_available = torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1"
except ImportError:
    cuda_available = False

ext_modules = []
cmdclass = {}

if cuda_available:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules = [
        CUDAExtension(
            name="mospork_kernels",
            sources=[
                "csrc/bindings.cpp",
                "csrc/kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    # Generate code for multiple GPU architectures:
                    #   Pascal (sm_60, sm_61), Volta (sm_70), Turing (sm_75),
                    #   Ampere (sm_80, sm_86), Hopper (sm_90)
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_90,code=sm_90",
                    "--ptxas-options=-v",
                ],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="musical-octo-spork",
    version="0.1.0",
    description="Model code and CUDA kernels for accelerated forward/backward passes",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=["torch>=2.0.0"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
