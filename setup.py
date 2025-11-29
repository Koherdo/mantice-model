from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mantice-model",
    version="1.0.0",
    author="Marcel NDEFFO, Hervé KOHO",
    author_email="marcel.ndeffo@tesselite.com",
    description="Superdiffusive Transport via Self-Organized Quaternionic Mantices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tesselite/mantice-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "nbsphinx",
        ],
    },
    entry_points={
        "console_scripts": [
            "mantice=src.cli:main",
        ],
    },
)