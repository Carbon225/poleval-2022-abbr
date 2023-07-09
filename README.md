# Abbreviation Disambiguation in Polish Press News Using Encoder-Decoder Models

## Summary

Abbreviation disambiguation is the process of expanding abbreviations, e.g. "eng." to the full form "engineer". In the Polish language the task is further complicated because of the many ways to create abbreviations and additional inflected forms. Abbreviation disambiguation was the topic of the 2022 PolEval Competition Task 2. This is the source code from the paper "Abbreviation Disambiguation in Polish Press News Using Encoder-Decoder Models" based on the competition.

## Usage

### Datasets

The datasets used in the paper are available in the `poleval_dataset.py` and `wiki_dataset.py` scripts. The Wikipedia pre-training dataset is also available on [HuggingFace](https://huggingface.co/datasets/carbon225/poleval-abbreviation-disambiguation-wiki).

### Training

The main `train.py` script can be used to train a model. The script takes a configuration file as an argument or a full list of hyperparameters. The configurations used in the paper are in the `configs` directory.

### Voting

The `voting.py` script can be used to perform majority voting on a set of predictions.

## Trained Models

The best trained models are available on HuggingFace:

* [plt5-wiki-train-dict](https://huggingface.co/carbon225/plt5-abbreviations-pl)
* [byt5-wiki-train-dict](https://huggingface.co/carbon225/byt5-abbreviations-pl)

## Installation

If using a custom virtualenv (not needed with a global Poetry installation):

```bash
pip install poetry
```

Then run:

```bash
poetry install
```
