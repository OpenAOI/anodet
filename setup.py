
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="image_anomaly_detection",
    version="0.0.1",
    author="OpenAOI",
    author_email="anton.emanuel@icloud.com",
    description="A set of functions for image anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/openaoi/padim_implementation",
    project_urls={
        "Source Code": "https://gitlab.com/openaoi/padim_implementation",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3',
)
