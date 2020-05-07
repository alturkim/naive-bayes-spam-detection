# Spam Detection using Naive Bayes
A naive bayes classifier is build to classify emails into two classes: Spam and Ham.

## Hyperparameters:
- clipped_count: While estimating the model parameter with Maximum Likelihood Estimation, this boolean parameter offers
possibility of using binary count of token, that is, a token count is incremented by one if it is found in an email,
regardless of how many time it is repeated in that email, in other words, count is clipped at one. Which helped enhance
performance on the dev set.
- alpha: smoothing constant. For Laplace smoothing set to 1.
 
## Implementation
The implementation is in pure python.

## Usage
`python naive_bayes.py train_dir test_dir --clipped_count --alpha 0.001`
