#!/usr/bin/env python3
"""
DuetMind Adaptive - Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_file(filename):
    """Read a file and return its contents"""
    with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements = []
    try:
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    except FileNotFoundError:
        pass
    return requirements

setup(
    name="duetmind_adaptive",
    version="1.0.0",
    description="Hybrid AI framework combining Adaptive Neural Networks with DuetMind cognitive agents",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    author="DuetMind Team",
    author_email="team@duetmind.ai",
    url="https://github.com/V1B3hR/duetmind_adaptive",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
        ],
        'viz': [
            'matplotlib>=3.4.0',
            'plotly>=5.0.0',
        ],
        'web': [
            'streamlit>=1.0.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'duetmind=duetmind_adaptive.main:main',
            'duetmind-train=duetmind_adaptive.training.cli:train_cli',
            'duetmind-api=duetmind_adaptive.api.server:run_server',
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="ai, machine learning, neural networks, medical ai, adaptive systems",
)