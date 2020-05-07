# Spam Detection using Naive Bayes
A naive bayes classifier is built to classify emails into two classes: Spam and Ham.

## Hyperparameters:
- clipped_count: While estimating the model parameters with Maximum Likelihood Estimation, this boolean parameter offers
the choice of using binary counts of tokens, that is, a token count is incremented by one if it is found in an email,
regardless of how many times it is repeated in that email, in other words, count is clipped at one, which helped enhancing the
performance on the dev set.
- alpha: smoothing constant. For Laplace smoothing set it to 1.
 
## Implementation
The implementation is in pure python 3.

## Usage
`python naive_bayes.py train_dir test_dir --clipped_count --alpha 0.001`
