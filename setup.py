#!/usr/bin/env python3
"""
Setup script for our BTC trading AI package.
"""

from setuptools import setup, find_packages

setup(
    name="btc-trading-ai",
    version="1.0.0",
    description="AI-powered Bitcoin trading signal classifier",
    author="BTC Trading Team", 
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0", 
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
        "google-cloud-storage>=2.10.0",
    ],
    entry_points={
        'console_scripts': [
            'train-btc-model=scripts.gcp.vertex_train_csv:main',
        ],
    },
)