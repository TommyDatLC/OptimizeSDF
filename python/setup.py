from setuptools import setup, find_packages

setup(
    name="optimize_sdf",
    version="1.0.0",
    description="GPU-accelerated Shape Diameter Function using NVIDIA OptiX RT Cores",
    author="USTH Final Project Team",
    packages=find_packages(),
    package_data={"optimize_sdf": ["lib/*.dll", "lib/*.ptx"]},
    include_package_data=True,
    install_requires=["numpy"],
    python_requires=">=3.8",
)
