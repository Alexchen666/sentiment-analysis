# Multi-Head Attention Sentiment Classifier Implementation

Build a transformer encoder from scratch for sentiment analysis using PyTorch. This project focuses on understanding and implementing the core components of transformer architecture.

## Data Source

IMDB Dataset of 50K Movie Reviews from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), which is originally from the publication Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).

The link to the paper: [Learning Word Vectors for Sentiment Analysis](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)

## Model Structure

The model consists of the following components:

1. **Embedding Layer**: Converts input tokens into dense vectors.
2. **Multi-Head Self-Attention**: Allows the model to focus on different parts of the input sequence.
3. **Positional Encoding**: Adds positional information to the input embeddings.
4. **Feed-Forward Neural Network**: Applies a non-linear transformation to the attention outputs.
5. **Layer Normalisation**: Stabilises and accelerates training.
6. **Output Layer**: Produces the final sentiment classification.

## Training and Evaluation

The model is trained using the AdamW optimiser and cross-entropy loss. The training process includes:
- Loading and preprocessing the IMDB dataset.
- Tokenising the text data.
- Training the model for a specified number of epochs.
- Evaluating the model on a test set.

## Result

```
Train Loss: 0.3027, Train Acc: 0.8699
Test Loss: 0.3056, Test Acc: 0.8663
```