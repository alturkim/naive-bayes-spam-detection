import glob
import os
import random
import math


def update_count_dict(emails, dictionary, clipped_count):
    """Update the passed tokens-count dictionary by adding the occurrences of tokens in the passed emails,
       counts are clipped to one per email if clipped_count is true.

    """
    if clipped_count:
        emails = [set(email) for email in emails]

    for email in emails:
        for token in email:
            dictionary[token] += 1


def get_subset(data, percentage, seed=1):
    """Returns a random subset of the data

    Args:
        data (list): Paths to .txt files
        percentage (float): Percentage of data to be returned
        seed (int): A seed to be used by the random number generator
                    (default is 1)
    Returns:
        list: Subset of paths to .txt files
    """
    subset_size = math.ceil(percentage*len(data))
    random.seed(seed)
    idxs = random.sample(range(len(data)), k=subset_size)
    return [data[idx] for idx in idxs]


def get_tokenized_corpus(files):
    """Read, preprocess and tokenize all training txt files

    Returns:
        list: list of ham emails, each email is a list of tokens
        list: list of spam emails, each email is a list of tokens
    """
    ham_emails = []
    spam_emails = []
    for file in files:
        if file.find('ham') >= 0:
            with open(file, "r", encoding="latin1") as f:
                ham_emails.append(f.read().lower().split())

        if file.find('spam') >= 0:
            with open(file, "r", encoding="latin1") as f:
                spam_emails.append(f.read().lower().split())

    return ham_emails, spam_emails


def read_dataset(path):
    search_str = os.path.join(path, '**/*.txt')
    files = glob.glob(search_str, recursive=True)
    ham_emails, spam_emails = get_tokenized_corpus(files)
    return ham_emails + spam_emails, ['ham' for _ in ham_emails] + ['spam' for _ in spam_emails]


def metrics(ys, ys_hat, positive='spam', negative='ham'):
    """Calculate precision, recall and f1 score for the positive class

    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for y, y_hat in zip(ys, ys_hat):
        if y_hat == positive:
            if y == positive:
                true_positive += 1
            else:
                false_positive += 1
        elif y_hat == negative:
            if y == negative:
                true_negative += 1
            else:
                false_negative += 1

        if true_positive == 0:
            precision, recall, f_1 = 0, 0, 0
        else:
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f_1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f_1
