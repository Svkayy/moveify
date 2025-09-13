#!/usr/bin/env python3
"""
Setup script for Dance Sync Analysis
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dance-sync-analysis",
    version="2.0.0",
    author="Dance Sync Team",
    author_email="dance-sync@example.com",
    description="Compare your dance moves to a model dancer's performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/dance-sync-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "dance-sync=dance:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.praat", "*.txt", "*.md"],
    },
)
