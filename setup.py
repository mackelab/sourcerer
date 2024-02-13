from setuptools import find_packages, setup

NAME = "sourcerer"
DESCRIPTION = "Sample-based Maximum Entropy Source Distribution Estimation"
URL = "NONE"
AUTHOR = "Julius Vetter, Guy Moss"
REQUIRES_PYTHON = ">=3.8.0"

REQUIRED = [
    "numpy",
    "torch",
    "matplotlib",
    "scikit-learn",
    "hydra-core",
    "pandas",
    "corner",
    "scipy",
    "brian2",
    "seaborn",
    "torchdiffeq",
]

EXTRAS = {
    "dev": [
        "autoflake",
        "black",
        "deepdiff",
        "flake8",
        "isort",
        "ipykernel",
        "jupyter",
        "pep517",
        "pytest",
        "pyyaml",
    ],
}

setup(
    name=NAME,
    version="0.1.0",
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="AGPLv3",
)
