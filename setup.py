from pathlib import Path
from setuptools import find_packages, setup


long_description = (Path(__file__).parent / "README.md").read_text()

core_requirements = [
    "gymnasium",
    "rich",
    "tqdm",
    "ipdb",
    "mujoco",
    "mujoco-mjx",
    "dm_control",
    "imageio",
    "gymnax",
    "brax==0.9.4",
    "torch",
    "opencv-python",
]

setup(
    name="humanoid_bench",
    version="0.1",
    author="RLL at UC Berkeley",
    url="https://github.com/carlosferrazza/humanoid-bench",
    description="Humanoid Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">3.7",
    install_requires=core_requirements,
)
