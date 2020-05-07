import argparse
import itertools
from utils import *


class NaiveBayes:
    def __init__(self, alpha, clipped_count):
        self.alpha = alpha
        self.clipped_count = clipped_count
        self.ham_prior = None
        self.spam_prior = None
        self.ham_tokens_likelihood_dict = None
        self.spam_tokens_likelihood_dict = None
        self.vocab_size = None

    def calculate_priors(self, labels):
        """Calculate MLE estimations of ham and spam priors

        """
        ham_count = 0
        spam_count = 0
        for label in labels:
            if label == 'ham':
                ham_count += 1
            else:
                spam_count += 1

        self.ham_prior = ham_count / (ham_count + spam_count)
        self.spam_prior = spam_count / (ham_count + spam_count)

    def calculate_likelihood_dicts(self, emails, labels):
        """Calculate MLE estimations of the conditional probabilities of each token in the vocabulary

        """
        ham_emails = [email for email, label in zip(emails, labels) if label=='ham']
        spam_emails = [email for email, label in zip(emails, labels) if label=='spam']

        all_ham_tokens = list(itertools.chain(*ham_emails))
        all_spam_tokens = list(itertools.chain(*spam_emails))

        num_ham_tokens = len(all_ham_tokens)
        num_spam_tokens = len(all_spam_tokens)
        self.vocab_size = sorted(set(list(all_ham_tokens) + list(all_spam_tokens)))

        ham_tokens_counts = dict(zip(self.vocab_size, [0 for _ in range(len(self.vocab_size))]))
        spam_tokens_counts = dict(zip(self.vocab_size, [0 for _ in range(len(self.vocab_size))]))

        update_count_dict(ham_emails, ham_tokens_counts, self.clipped_count)
        update_count_dict(spam_emails, spam_tokens_counts, self.clipped_count)

        self.ham_tokens_likelihood_dict = {k: (v + self.alpha) / (num_ham_tokens + self.alpha * len(self.vocab_size))
                                           for k, v in ham_tokens_counts.items()}
        self.spam_tokens_likelihood_dict = {k: (v + self.alpha) / (num_spam_tokens + self.alpha * len(self.vocab_size))
                                            for k, v in spam_tokens_counts.items()}

    def log_prob_of_spam(self, email):
        """calculate log p(email | spam)

        Args:
            email (list): list of tokens

        Returns:
            float: p(email | spam)
        """
        log_p_email_given_spam = 0
        for token in email:
            try:
                log_p_email_given_spam += math.log(self.spam_tokens_likelihood_dict[token])
            except KeyError:
                # ignore the token if it did not appear in training data
                pass
        return log_p_email_given_spam + math.log(self.spam_prior)

    def log_prob_of_ham(self, email):
        """calculate log p(email | ham)

        Args:
            email (list): list of tokens

        Returns:
            float: p(email | ham)
        """
        log_p_email_given_ham = 0
        for token in email:
            try:
                log_p_email_given_ham += math.log(self.ham_tokens_likelihood_dict[token])
            except KeyError:
                pass
        return log_p_email_given_ham + self.ham_prior

    def predict_one_email(self, email):
        """classify email as spam or ham

        Args:
            email (list): list of tokens

        Returns:
            str: predicted class of the email

        """
        if self.clipped_count:
            email = set(email)
        try:
            return 'spam' if self.log_prob_of_spam(email) >= self.log_prob_of_ham(email) else 'ham'
        except TypeError:
            raise TypeError('ERROR: The model is not trained, call NaiveBayes.fit() first')

    def fit(self, emails, labels):
        self.calculate_priors(labels)
        self.calculate_likelihood_dicts(emails, labels)

    def predict(self, emails):
        return [self.predict_one_email(email) for email in emails]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir')
    parser.add_argument('test_dir')
    parser.add_argument('--clipped_count', action='store_true')
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()
    train_dir = args.train_dir
    test_dir = args.test_dir
    clipped_count = args.clipped_count
    alpha = args.alpha

    train_emails, train_labels = read_dataset(train_dir)
    test_emails, test_labels  = read_dataset(test_dir)

    nb = NaiveBayes(alpha=alpha, clipped_count=clipped_count)
    nb.fit(train_emails, train_labels)
    ys_hat = nb.predict(test_emails)

    print(metrics(test_labels, ys_hat))
