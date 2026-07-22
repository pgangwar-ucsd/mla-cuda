from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# We enable O3 optimization and target modern Ampere/Hopper architectures
extra_compile_args = {
    'cxx': ['-O3', '-Wall'],
    'nvcc': [
        '-O3',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-gencode=arch=compute_80,code=sm_80', # A100 / RTX 30-series
        '-gencode=arch=compute_89,code=sm_89', # RTX 40-series / Ada
        '-gencode=arch=compute_90,code=sm_90', # H100
    ]
}

setup(
    name='mla_custom_cuda',
    ext_modules=[
        CUDAExtension(
            name='mla_custom_cuda',
            sources=['naive_fused_kernel_your_playground.cu'],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)