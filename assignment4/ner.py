from nltk.corpus import conll2002
import numpy as np
import string
from sklearn import svm
from sklearn.model_selection import learning_curve
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support

# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, pos, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    features = [
        (o + '-word', word),
        (o + '-pos', pos),
        (o + '-istitle', word.istitle()),
        (o + '-isdigit', word.isdigit()),
        (o + '-hasdigit', any(char.isdigit() for char in word)),
        (o + '-bias', int(o)),
        (o + '-isupper', word.isupper()),
        (o + '-hashyphen', ('-' in word)),
        (o + '-haspercent', ('%' in word)),
        (o + '-allcaps', word.isupper()),
        (o + '-haspunct', any([c in string.punctuation for c in word])),
        (o + '-ispunct', all([c in string.punctuation for c in word])),
        (o + '-prefix1', word[:1]),
        (o + '-prefix2', word[:2]),
        (o + '-prefix3', word[:3]),
        (o + '-suffix1', word[-1:]),
        (o + '-suffix2', word[-2:]),
        (o + '-suffix3', word[-3:])

        # TODO: add more features here.
    ]
    return features
    

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            pos = sent[i+o][1]
            featlist = getfeats(word, pos, o)
            features.extend(featlist)
    d = dict(features)
    return d


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))
    
    train_feats = []
    train_labels = []

    poses = []
    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])
            poses.append(sent[i][1])

    # print(set(poses))

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    train_labels = np.asarray(train_labels)

    # TODO: play with other models
    # BEST CONSTRAINED MODEL

    model = Perceptron(
        alpha=0.00001,
        max_iter=100,
        verbose=1
    )

    # model = MLPClassifier(
    #     solver='sgd',
    #     alpha=0.00001,
    #     activation='relu',
    #     max_iter=100,
    #     verbose=1
    # )

    # model = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs',
    #     c1=0.1,
    #     c2=0.1,
    #     max_iterations=100,
    #     all_possible_states=True
    # )

    # BEST UNCONSTRAINED MODEL

    # model = SGDClassifier(
    #     loss='hinge',
    #     alpha=0.00001,
    #     max_iter=100
    # )

    # model = PassiveAggressiveClassifier(
    #     max_iter=100,
    #     loss='hinge'
    # )

    # model = svm.SVC(
    #     kernel='rbf',
    #     max_iter=-1,
    #     verbose=True
    # )

    # model = svm.LinearSVC(
    #     multi_class='crammer_singer'
    # )

    model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in dev_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("constrained_results.txt", "w") as out:
        for sent in dev_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")






