# SC4002 Natural Language Processing Project

Group 63 project for NTU SC4002 (Natural Language Processing), using the TREC
question-classification dataset to examine static word embeddings, OOV handling,
and neural text-classification architectures.

## Project overview

The task classifies English questions into six coarse-grained categories
(`fine_grained=False`):

- `DESC` - description
- `ENTY` - entity
- `ABBR` - abbreviation
- `HUM` - human
- `NUM` - number
- `LOC` - location

The TorchText pipeline uses spaCy (`en_core_web_sm`) tokenization and
100-dimensional GloVe embeddings (`glove.6B.100d`), yielding a vocabulary of
9,330 entries (including `<unk>` and `<pad>`). Splitting the training set 85/15
(`random.seed(42)`) produces **4,634 training**, **818 validation**, and
**500 test** examples.

## Repository structure

```text
.
├── Part 1/
│   ├── SC4002_Part1.ipynb       # Vocabulary, OOV analysis, and embedding export
│   ├── word_embeddings.npy      # Exported GloVe embedding matrix
│   └── word_list.npy            # Vocabulary list aligned with word_embeddings.npy
├── Part 2/
│   └── Part2.ipynb              # Baseline 1-layer BiLSTM classifier
├── Part 3/
│   ├── Part3a.ipynb             # 2-layer BiLSTM and BiGRU comparison
│   ├── Part3b.ipynb             # TextCNN classifier
│   ├── Part3c.ipynb             # BERT (bert-base-uncased) classifier
│   └── Part3d.ipynb             # Class-weighted BiGRU with early stopping
├── SC4002 Report G63.pdf        # Full project report
└── README.md
```

## Experiments and results

Each model is selected by best validation accuracy and evaluated on the
500-example TREC test set:

| Experiment | Model | Key setup | Test loss | Test accuracy |
| --- | --- | --- | ---: | ---: |
| [Part 2](Part%202/Part2.ipynb) | BiLSTM (baseline) | 1 layer, hidden 128, dropout 0.5, clip 1.0, patience 5 | 0.3916 | 90.61% |
| [Part 3(a)](Part%203/Part3a.ipynb) | BiLSTM | 2 layers, hidden 256, 20 epochs | 0.278 | 92.35% |
| [Part 3(a)](Part%203/Part3a.ipynb) | BiGRU | 2 layers, hidden 256, 20 epochs | 0.231 | 93.22% |
| [Part 3(b)](Part%203/Part3b.ipynb) | TextCNN | 100 filters x kernels `[3, 4, 5]`, 20 epochs | 0.387 | 88.46% |
| [Part 3(c)](Part%203/Part3c.ipynb) | BERT (`bert-base-uncased`) | `lr=2e-5`, batch 16, `max_len=128` | 0.160 | 96.80% |
| [Part 3(d)](Part%203/Part3d.ipynb) | Class-weighted BiGRU | 2 layers, hidden 256, `balanced` weights, patience 5 | 0.274 | 92.74% |

*Values above reflect the saved notebook outputs. In the TorchText notebooks,
reported test accuracy averages per-batch accuracy; earlier/report runs in
[SC4002 Report G63.pdf](SC4002%20Report%20G63.pdf) differ slightly for the
non-BERT models.*

### Key findings

- **OOV mitigation ([Part 1](Part%201/SC4002_Part1.ipynb)):** Dropping the
  initial `min_freq=3` threshold expanded the training vocabulary from 2,265 to
  9,330 entries and eliminated observed OOV tokens (7,065 to 0).
- **Model comparison (Part 3a-c):** The 2-layer BiGRU (93.22%) outperformed the
  2-layer BiLSTM (92.35%) and TextCNN (88.46%), while fine-tuning
  `bert-base-uncased` reached 96.80% (macro F1 0.95) via contextualized
  representations and WordPiece subwords.
- **Class weighting trade-off ([Part 3d](Part%203/Part3d.ipynb)):**
  Inverse-frequency class weights (`compute_class_weight('balanced')`) lifted
  recall on moderately imbalanced classes such as `ENTY` (83% to 88%) and `HUM`
  (94% to 97%) in the report's before/after comparison, while `ABBR` recall
  remained 67% (6/9 test examples) due to severe data scarcity.

## Requirements and usage

The notebooks target Python 3 (Jupyter or Google Colab). Key dependencies:

- PyTorch and `torchtext==0.4.0`
- spaCy (`python -m spacy download en_core_web_sm`)
- NumPy, Matplotlib, NLTK, scikit-learn
- `transformers` (Part 3(c) only)

Notes on running:

- Each notebook is **self-contained**: Parts 2 and 3 rebuild the TREC/GloVe
  pipeline independently rather than loading `Part 1/*.npy`, though Part 1 to
  Part 3 is the intended reading order.
- Because `torchtext==0.4.0` uses legacy `torchtext.data` APIs, Colab or an
  isolated Python environment pinned to `torchtext==0.4.0` is recommended.
- First execution downloads the TREC dataset and `glove.6B.zip` (~862 MB); a
  CUDA runtime is recommended for Part 3 (especially BERT).
- Checkpoints (`.pt`) and training-curve plots (`.png`) written during training
  are runtime artifacts and are not required to inspect the notebooks.

## Report and contributors

See [SC4002 Report G63.pdf](SC4002%20Report%20G63.pdf) for full methodology,
hyperparameter and regularization ablations, training curves, and error
analysis.

| Member | Scope |
| --- | --- |
| Dohyun Jeong | Part 1, Report |
| Tan Kuan Kiat, Chua Shun Kah | Part 2, Report |
| Goh Jie Rong, Sean; Rana Khushi | Part 3, Report |
