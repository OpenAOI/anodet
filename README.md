# image_anomaly_detection

A set of functions and classes for performing anomaly detection in images using features from pretrained neural networks.

The package includes functions and classes for extracting, modifying and comparing features. It also includes unofficial implementations of [**PaDiM**](https://arxiv.org/abs/2011.08785) and [**PatchCore**](https://arxiv.org/abs/2106.08265).

Some code has been borrowed and/or inspired by other repositories, see code reference below.


## Installation

Clone the repository
```
git clone https://gitlab.com/openaoi/padim_implementation.git
```

Install the package

```
cd padim_implementation
python -m pip install -r requirements.txt
python -m pip install .
```


## Usage example

See [notebooks](https://gitlab.com/openaoi/padim_implementation/-/tree/master/notebooks) for in depth examples.


## Development setup

#### Install

Install the package in editable mode
```
pip3 install --editable [PATH TO REPOSITORY]
```

#### Tests

Install packages for testing
```
python -m pip install pytest pytest-mypy pytest-flake8
```

Run tests
```
cd [PATH TO REPOSITORY]
pytest --mypy --flake8
```

For configuration of pytest, mypy and flake8 edit `setup.cfg`.


#### Creating docs

Install pydoc-markdown
```
python -m pip install pydoc-markdown
```

Clone docs repository
```
git clone https://gitlab.com/openaoi/padim_implementation.wiki.git
```

Run script
```
cd padim_implementation.wiki
python generate_docs.py --source-path=[PATH TO REPOSITORY] --package-name="image_anomaly_detection" --save-path=.
```




## Code Reference

Some parts used in patch_core.py :
https://github.com/hcw-00/PatchCore_anomaly_detection

Code in directory sampling_methods :
https://github.com/google/active-learning

concatenate_two_layers function in feature_extraction.py :
https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

pytorch_cov function in utils.py :
https://github.com/pytorch/pytorch/issues/19037




