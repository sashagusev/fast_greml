from setuptools import Extension, setup

import numpy as np


setup(
    name="fast_greml",
    version="0.1.0",
    ext_modules=[
        Extension(
            "_greml_accel",
            sources=["_greml_accel.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
            extra_link_args=["-framework", "Accelerate"],
        )
    ],
)
