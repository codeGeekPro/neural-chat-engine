#!/usr/bin/env python3
"""
Neural Chat Engine - Setup Configuration
Chatbot IA Avancé avec Deep Learning et NLP
"""

from setuptools import setup, find_packages
import os

# Lire le README pour la description longue
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Lire les requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neural-chat-engine",
    version="0.1.0",
    author="CodeGeekPro",
    author_email="contact@codegeekpro.com",
    description="Chatbot IA Avancé avec Deep Learning, NLP et Capacités Multimodales",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/codeGeekPro/neural-chat-engine",
    project_urls={
        "Bug Tracker": "https://github.com/codeGeekPro/neural-chat-engine/issues",
        "Documentation": "https://github.com/codeGeekPro/neural-chat-engine/docs",
        "Source Code": "https://github.com/codeGeekPro/neural-chat-engine",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
            "torchaudio>=2.0.0+cu118",
            "faiss-gpu>=1.7.0",
        ],
        "production": [
            "gunicorn>=21.0.0",
            "prometheus-client>=0.17.0",
            "sentry-sdk>=1.28.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
            "torchaudio>=2.0.0+cu118",
            "faiss-gpu>=1.7.0",
            "gunicorn>=21.0.0",
            "prometheus-client>=0.17.0",
            "sentry-sdk>=1.28.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Communications :: Chat",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords=[
        "chatbot", "ai", "nlp", "deep-learning", "transformer", 
        "pytorch", "langchain", "conversation", "neural-network",
        "machine-learning", "artificial-intelligence", "rag"
    ],
    entry_points={
        "console_scripts": [
            "neural-chat=neural_chat_engine.cli:main",
            "nce-train=neural_chat_engine.training.cli:main",
            "nce-serve=neural_chat_engine.api.serve:main",
        ],
    },
    include_package_data=True,
    package_data={
        "neural_chat_engine": [
            "configs/*.yaml",
            "configs/*.json",
            "models/*.json",
            "data/samples/*.json",
        ],
    },
    zip_safe=False,
)