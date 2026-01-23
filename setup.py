from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# Read requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        # Filter out nvidia packages that may not be available on all platforms
        requirements = [req for req in requirements if not req.startswith('nvidia-')]
else:
    requirements = []

setup(
    name="alpharl",
    version="0.1.0",
    author="Sergiy Sanin",
    author_email="sanin_sergiy@yahoo.com",
    description="AlphaRL: Reinforcement Learning Trading Strategy Simulator using PPO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alpharl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=1.0.0",
        "stable-baselines3>=2.0.0",
        "numpy>=1.20.0",
        "pandas>=2.0.0",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
