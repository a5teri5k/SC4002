# SC4002 Natural Language Processing Project

This repository contains the SC4002 Natural Language Processing group project. The
project uses the TREC question-classification dataset to investigate word
embeddings and compare several neural text-classification models.

## Project overview

The experiments classify questions into six coarse-grained categories:

- `DESC` - description
- `ENTY` - entity
- `ABBR` - abbreviation
- `HUM` - human
- `NUM` - number
- `LOC` - location

The notebooks use the TorchText TREC dataset, spaCy tokenization, and
100-dimensional GloVe embeddings (`glove.6B.100d`). The resulting vocabulary
contains 9,330 entries, including the special tokens used by the pipeline.

## Repository structure

```text
.
├── Part 1/
│   ├── SC4002_Part1.ipynb       # Vocabulary, OOV, and embedding analysis
│   ├── word_embeddings.npy      # Exported embedding matrix
│   └── word_list.npy            # Vocabulary corresponding to the matrix
├── Part 2/
│   └── Part2.ipynb              # Baseline classifier and evaluation
├── Part 3/
│   ├── Part3a.ipynb             # BiLSTM and BiGRU comparison
│   ├── Part3b.ipynb             # CNN classifier
│   ├── Part3c.ipynb             # BERT classifier
│   └── Part3d.ipynb             # Class-weighted BiGRU with early stopping
├── SC4002 Report G63.pdf        # Project report
└── README.md
```

## Experiments and recorded results

The notebooks save the best validation checkpoint and report performance on the
500-example TREC test set:

| Experiment | Model | Test accuracy |
| --- | --- | ---: |
| Part 2 | TorchText baseline classifier | 90.61% |
| Part 3(a) | BiLSTM | 92.35% |
| Part 3(a) | BiGRU | 93.22% |
| Part 3(b) | CNN | See notebook output |
| Part 3(c) | BERT (`bert-base-uncased`) | 96.80% |
| Part 3(d) | Class-weighted BiGRU with early stopping | 92.74% |

Part 1 also investigates out-of-vocabulary (OOV) words and visualizes
topic-specific word embeddings. Removing the minimum-frequency restriction
used by the initial vocabulary configuration eliminates the observed OOV words
in the resulting training vocabulary.

## Requirements

The notebooks were authored for Python 3 and can be run in Jupyter Notebook or
Google Colab. The main dependencies are:

- PyTorch
- `torchtext==0.4.0`
- spaCy with the `en_core_web_sm` model
- NumPy
- Matplotlib
- NLTK
- `transformers` (Part 3(c) only)
- scikit-learn (used for evaluation/visualization in relevant notebooks)

The notebooks include installation cells for several dependencies. Because
`torchtext==0.4.0` is an older release, running the notebooks is most reliable
in the Colab environment for which they were prepared. A CUDA-enabled runtime
is recommended for the Part 3 experiments, especially BERT.

## Running the notebooks

1. Clone or download this repository.
2. Open the required notebook in Jupyter or Google Colab.
3. Run the cells from top to bottom.
4. Allow the notebooks to download the TREC dataset and GloVe vectors when
   prompted.

Run the notebooks in this order when reproducing the complete project:

1. `Part 1/SC4002_Part1.ipynb` - prepare and inspect the vocabulary and
   embeddings.
2. `Part 2/Part2.ipynb` - train the baseline classifier.
3. `Part 3/Part3a.ipynb` through `Part 3/Part3d.ipynb` - train and compare the
   remaining models.

Some experiments generate model checkpoints and training-curve plots in the
notebook's working directory. These generated files are not required to inspect
the source notebooks.

## Report

See [SC4002 Report G63.pdf](SC4002%20Report%20G63.pdf) for the full methodology,
analysis, plots, and discussion of the results.
