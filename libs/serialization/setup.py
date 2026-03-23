from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# compile for all possible CUDA architectures
# all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()
# alternatively, you can list cuda archs that you want, eg:
# check https://developer.nvidia.com/cuda-gpus to find your arch
all_cuda_archs = [
    # '-gencode', 'arch=compute_90,code=sm_90',
    # '-gencode', 'arch=compute_75,code=sm_75',
    # '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86'
]

setup(
    name='serialization_cuda',
    ext_modules=[
        CUDAExtension(
            name='serialization_cuda',
            sources=[
                'cuda/bindings.cpp',
                'cuda/hilbert.cu',
                'cuda/morton.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']+all_cuda_archs
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)