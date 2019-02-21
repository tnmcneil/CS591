#############################################################
# ASSIGNMENT 1
# Theresa McNeil
# U09757615
# RELEASED: 2/6/2019
# DUE: 2/15/2019
# DESCRIPTION: In this assignment, you will explore the
# text classification problem of identifying complex words.
#############################################################

from collections import defaultdict
import gzip
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#### 1. Evaluation Metrics ####

# Input: y_pred, a list of length n with the predicted labels,
# y_true, a list of length n with the true labels
# complex words are positive (1) and simple words are negative (0)

# Calculate true positives, false positives, true negatives and false negatives
def get_metrics(y_pred, y_true):
    n = len(y_pred)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(n):
        if (y_pred[i] == 1) and (y_true[i] == 1):
            tp += 1
        elif (y_pred[i] == 1) and (y_true[i] == 0):
            fp += 1
        elif (y_pred[i] == 0) and (y_true[i] == 0):
            tn += 1
        elif (y_pred[i] == 0) and (y_true[i] == 1):
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

# Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    precision, recall = get_metrics(y_pred, y_true)
    return precision

# Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    precision, recall = get_metrics(y_pred, y_true)
    return recall

# Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    precision, recall = get_metrics(y_pred, y_true)
    fscore = 2*(precision*recall)/(precision+recall)
    return fscore

# returns the calculated recall, precision and recall
def evaluate(y_pred, y_true):
    precision, recall = get_metrics(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    return precision, recall, fscore


#### 2. Complex Word Identification ####

# Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


# 2.1: A very simple baseline
# Makes feature matrix for all complex
def all_complex_feature(words):
    n = len(words)
    pred = [1]*n
    return pred

# Labels every word complex
def all_complex(data_file):
    words, labels = load_file(data_file)
    pred = all_complex_feature(words)
    precision, recall, fscore = evaluate(pred, labels)
    performance = [precision, recall, fscore]
    return performance


# 2.2: Word length thresholding
# Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    n = len(words)
    pred = [None]*n
    for i in range(n):
        if len(words[i]) >= threshold:
            pred[i] = 1
        else:
            pred[i] = 0
    return pred


# Finds the best length threshold by f-score, and uses this threshold to
# classify the training and development set
def word_length_threshold(training_file, development_file):
    THRESHHOLD = 8

    twords, tlabels = load_file(training_file)
    dwords, dlabels = load_file(development_file)
    tpred = length_threshold_feature(twords, THRESHHOLD)
    dpred = length_threshold_feature(dwords, THRESHHOLD)

    tprecision, trecall, tfscore = evaluate(tpred, tlabels)
    dprecision, drecall, dfscore = evaluate(dpred, dlabels)

    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


# 2.3: Word frequency thresholding

# Loads Google NGram counts
def load_ngram_counts(ngram_counts_file):
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt', encoding='iso8859') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
# classify the training and development set


# Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    n = len(words)
    pred = [None]*n
    for i in range(n):
        word = words[i]
        if counts[word] <= threshold:
            pred[i] = 1
        else:
            pred[i] = 0
    return pred


def word_frequency_threshold(training_file, development_file, counts):
    THRESHHOLD = 11000000
    twords, tlabels = load_file(training_file)
    dwords, dlabels = load_file(development_file)
    tpred = frequency_threshold_feature(twords, THRESHHOLD, counts)
    dpred = frequency_threshold_feature(dwords, THRESHHOLD, counts)

    tprecision, trecall, tfscore = evaluate(tpred, tlabels)
    dprecision, drecall, dfscore = evaluate(dpred, dlabels)

    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


# 2.4: Naive Bayes
# Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    # load in training and dev files, get features & standardize
    twords, Y_t = load_file(training_file)
    X_train = np.array([[counts[word], len(word)] for word in twords])
    dwords, Y_d = load_file(development_file)
    X_dev = np.array([[counts[word], len(word)] for word in dwords])
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    # X_dev = (X_dev - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    # build model trained on training data
    clf = GaussianNB()
    clf.fit(X_train, Y_t)
    # predict labels for training and development & get metrics
    Y_tpred = clf.predict(X_train).tolist()
    tprecision, trecall, tfscore = evaluate(Y_tpred, Y_t)
    Y_dpred = clf.predict(X_dev).tolist()
    dprecision, drecall, dfscore = evaluate(Y_dpred, Y_d)

    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


# 2.5: Logistic Regression
# Trains a Logistic Regression classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    # load in training and dev files, get features & standardize
    twords, Y_t = load_file(training_file)
    X_train = np.array([[counts[word], len(word)] for word in twords])
    dwords, Y_d = load_file(development_file)
    X_dev = np.array([[counts[word], len(word)] for word in dwords])
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    # X_dev = (X_dev - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    # build model
    clf = LogisticRegression()
    clf.fit(X_train, Y_t)
    # predict labels for training and development sets & get metrics
    Y_tpred = clf.predict(X_train).tolist()
    tprecision, trecall, tfscore = evaluate(Y_tpred, Y_t)
    Y_dpred = clf.predict(X_dev).tolist()
    dprecision, drecall, dfscore = evaluate(Y_dpred, Y_d)

    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

# 2.7: Build your own classifier

def random_forrest(training_file ,development_file, counts):
    twords, Y_t = load_file(training_file)
    X_train = np.array([[counts[word], len(word)] for word in twords])
    dwords, Y_d = load_file(development_file)
    X_dev = np.array([[counts[word], len(word)] for word in dwords])
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_dev = (X_dev - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_t)
    Y_tpred = clf.predict(X_train).tolist()
    tprecision, trecall, tfscore = evaluate(Y_tpred, Y_t)
    Y_dpred = clf.predict(X_dev).tolist()
    dprecision, drecall, dfscore = evaluate(Y_dpred, Y_d)

    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

# Trains a classifier of your choosing, predicts labels for the test dataset
# and writes the predicted labels to the text file 'test_labels.txt',
# with ONE LABEL PER LINE


def main():
    print("all results in form [precision, recall, F1]")
    print("results for training file on all complex baseline:")
    print(all_complex(training_file))
    print("results for development file on all complex baseline:")
    print(all_complex(development_file))

    train_result, dev_result = word_length_threshold(training_file, development_file)
    print("results for training file on word length baseline with threshhold 8")
    print(train_result)
    print("results for development file on word length baseline with threshhold 8")
    print(dev_result)

    train_result, dev_result = word_frequency_threshold(training_file, development_file, counts)
    print("results for training file on frequency baseline with threshhold 11mil")
    print(train_result)
    print("results for development file on frequency baseline with threshhold 11mil")
    print(dev_result)

    train_result, dev_result = naive_bayes(training_file, development_file, counts)
    print("results for training file on naive bayes, trained with train file")
    print(train_result)
    print("result for development file on naive bayes, trained with train file")
    print(dev_result)

    train_result, dev_result = logistic_regression(training_file, development_file, counts)
    print("results for training file on logistic regression, trained with train file")
    print(train_result)
    print("results for development file on logistic regression, trained with train file")
    print(dev_result)

    train_result, dev_result = random_forrest(training_file, development_file, counts)
    print("results for training file on random forrest, trained with train file")
    print(train_result)
    print("results for development file on random forrest, trained with train file")
    print(dev_result)


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)

    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    main()