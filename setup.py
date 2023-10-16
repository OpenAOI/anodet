from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="anodet",
    version="0.0.1",
    author="OpenAOI",
    author_email="anton.emanuel@icloud.com",
    description="A set of functions and classes for performing anomaly detection in \
                    images using features from pretrained neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/openaoi/anodet",
    project_urls={
        "Source Code": "https://gitlab.com/openaoi/anodet",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3",
)
