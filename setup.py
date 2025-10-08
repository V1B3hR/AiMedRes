#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AiMedRes - Setup Configuration

This file defines the package metadata and installation instructions for the
AimedRes project.

Note on Modern Packaging:
While setup.py is still supported, the Python packaging ecosystem is moving
towards declarative configurations in `pyproject.toml` and `setup.cfg`.
This script is a robust, updated version of the traditional setup.py.
"""

import os
import re
from setuptools import setup, find_packages
from typing import List

# ==============================================================================
# METADATA
# ==============================================================================
NAME = "aimedres"
DESCRIPTION = "AiMedRes - AI Medical Research Assistant for advanced healthcare analytics"
AUTHOR = "VIBEHR"  # Or your GitHub username/organization
AUTHOR_EMAIL = "your-email@example.com"  # A contact email
URL = "https://github.com/V1B3hR/aimedres"
LICENSE = "Apache Software License"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_version(package: str) -> str:
    """
    Dynamically reads the version string from the package's __init__.py.
    This establishes a single source of truth for the version.
    """
    init_py = open(os.path.join("src", package, "__init__.py")).read()
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", init_py, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError(f"Unable to find version string in src/{package}/__init__.py.")


def get_long_description(filename: str = "README.md") -> str:
    """
    Reads the README file and returns its contents for the long description.
    Falls back to the short description if the file is not found.
    """
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using short description.")
        return DESCRIPTION


def get_requirements(filename: str = "requirements.txt") -> List[str]:
    """
    Reads requirements from a file, ignoring comments and empty lines.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Warning: {filename} not found. No base requirements installed.")
        return []

# ==============================================================================
# DEPENDENCY DEFINITIONS
# ==============================================================================

# Base requirements for the package to function
INSTALL_REQUIRES = get_requirements()

# Dependencies for developers (testing, linting, formatting)
DEV_REQUIRES = [
    'pytest>=7.0.0',
    'pytest-cov>=3.0.0',
    'black>=22.0.0',
    'flake8>=4.0.0',
    'mypy>=0.900',
]

# Dependencies for data visualization
VIZ_REQUIRES = [
    'matplotlib>=3.5.0',
    'plotly>=5.5.0',
    'seaborn>=0.11.0',
]

# Dependencies for running a web interface (e.g., Streamlit demo)
WEB_REQUIRES = [
    'streamlit>=1.10.0',
]

# Dependencies for medical imaging analysis
IMAGING_REQUIRES = [
    'nibabel>=5.0.0',       # NIfTI file format support
    'pydicom>=2.4.0',      # DICOM file format support
    'SimpleITK>=2.3.0',    # Medical image processing (fixed name)
    'pyradiomics>=3.0.1',  # Radiomics feature extraction
    'nipype>=1.8.0',       # Neuroimaging pipelines
    'nilearn>=0.10.0',     # Neuroimaging machine learning
    'bids-validator>=1.13.0',  # BIDS compliance validation
    'pybids>=0.15.0',      # BIDS dataset management
]

# ==============================================================================
# SETUP CONFIGURATION
# ==============================================================================

setup(
    name=NAME,
    version=get_version(NAME),
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    
    # --- Project Structure ---
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True, # To include non-python files specified in MANIFEST.in
    
    # --- Dependencies ---
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'dev': DEV_REQUIRES,
        'viz': VIZ_REQUIRES,
        'web': WEB_REQUIRES,
        'imaging': IMAGING_REQUIRES,
        'all': DEV_REQUIRES + VIZ_REQUIRES + WEB_REQUIRES + IMAGING_REQUIRES,
    },

    # --- Command-line Scripts ---
    entry_points={
        'console_scripts': [
            'aimedres=aimedres.cli.commands:main',
            'aimedres-train=aimedres.cli.train:main',
            'aimedres-serve=aimedres.cli.serve:main',
        ],
    },
    
    # --- PyPI Metadata ---
    project_urls={
        "Bug Tracker": f"{URL}/issues",
        "Source Code": URL,
        "Documentation": f"{URL}/wiki", # Example link
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai, machine-learning, deep-learning, medical-imaging, healthcare, python",
)
