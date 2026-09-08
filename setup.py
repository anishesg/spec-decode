from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

src_dir = os.path.join(os.path.dirname(__file__), "src")
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

extra_cuda_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--generate-code=arch=compute_80,code=sm_80",
    "--generate-code=arch=compute_86,code=sm_86",
    "--generate-code=arch=compute_89,code=sm_89",
    "--generate-code=arch=compute_90,code=sm_90",
]

extra_cxx_flags = ["-O3", "-std=c++17"]

setup(
    name="spec_decode",
    version="0.1.0",
    description="Fused GPU-resident speculative decoding verification",
    packages=["spec_decode"],
    ext_modules=[
        CUDAExtension(
            name="spec_decode._C",
            sources=[
                os.path.join(csrc_dir, "bindings.cpp"),
                os.path.join(src_dir,  "reference.cu"),
                os.path.join(src_dir,  "fused_verify.cu"),
                os.path.join(src_dir,  "tree_verify.cu"),
            ],
            include_dirs=[src_dir],
            extra_compile_args={
                "cxx":  extra_cxx_flags,
                "nvcc": extra_cuda_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
