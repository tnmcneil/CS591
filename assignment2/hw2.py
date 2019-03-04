import os
import csv
import subprocess
import re
import random
import numpy as np
import math

def read_in_shakespeare():
    """
    Reads in the Shakespeare dataset processes it into a list of tuples.
    Also reads in the vocab and play name lists from files.

    Each tuple consists of
      tuple[0]: The name of the play
      tuple[1] A line from the play as a list of tokenized words.

    Returns:
      tuples: A list of tuples in the above format.
      document_names: A list of the plays present in the corpus.
      vocab: A list of all tokens in the vocabulary.
    """

    tuples = []

    with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((play_name, line_tokens))

    with open('vocab.txt') as f:
        vocab = [line.strip() for line in f]

    with open('play_names.txt') as f:
        document_names = [line.strip() for line in f]

    return tuples, document_names, vocab

def get_row_vector(matrix, row_id):
    return matrix[row_id, :]

def get_column_vector(matrix, col_id):
    return matrix[:, col_id]

def create_term_document_matrix(line_tuples, document_names, vocab):
    """
    Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      document_names: A list of the document names
      vocab: A list of the tokens in the vocabulary

    Let m = len(vocab) and n = len(document_names).

    Returns:
      td_matrix: A mxn numpy array where the number of rows is the number of words
      and each column corresponds to a document. A_ij contains the
      frequency with which word i occurs in document j.
    """

    # dictionary with key = vocab word and value = index at which it appears in vocab list
    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    # dictionary with key = doc name and value = index at which it appears in doc name list
    docname_to_id = dict(zip(document_names, range(0, len(document_names))))

    m = len(vocab)
    n = len(document_names)

    td_matrix = np.zeros((m,n))

    for i in range(len(line_tuples)):
        doc = line_tuples[i][0]
        line = line_tuples[i][1]
        d_id = docname_to_id[doc]
        for token in line:
            t_id = vocab_to_id[token]
            td_matrix[t_id][d_id] += 1

    # print(td_matrix[(vocab_to_id['soldier'])][17])
    # print(td_matrix[(vocab_to_id['soldier'])][28])
    # print(td_matrix[(vocab_to_id['soldier'])][18])
    # print(td_matrix[(vocab_to_id['soldier'])][26])
    # juliet = vocab_to_id['juliet']
    # romeo = vocab_to_id['romeo']
    # print("term doc")
    # print(td_matrix[juliet])
    # print(td_matrix[romeo])
    # print(np.sum(td_matrix[vocab_to_id['love']]))

    return td_matrix


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    """
    Returns a numpy array containing the term context matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      vocab: A list of the tokens in the vocabulary
      context_window_size: size of the word window around the target word

    Let n = len(vocab).

    Returns:
      tc_matrix: A nxn numpy array where A_ij contains the frequency with which
      word j was found within context_window_size to the left or right of
      word i in any sentence in the tuples.
    """

    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    n = len(vocab)
    tc_matrix = np.zeros((n,n))

    for i in range(0,len(line_tuples)):
        line = line_tuples[i][1]
        for j in range(0,len(line)):
            t_id = vocab_to_id[line[j]]
            lb = max(0, j-context_window_size)
            ub = min(len(line), j+context_window_size)
            context = line[lb:ub+1]
            for k in range(0,len(context)):
                # print("target word j:")
                # print(line[j])
                # print("context word i:")
                # print(context[k])
                c_id = vocab_to_id[context[k]]
                tc_matrix[t_id][c_id] += 1
                # print("co-occurrence of words " + str(t_id) + " and " + str(+ c_id))
                # print(tc_matrix[t_id][c_id])
    #
    # juliet = vocab_to_id['juliet']
    # romeo = vocab_to_id['romeo']
    # love = vocab_to_id['love']
    # print("Term context")
    # print(tc_matrix[juliet][juliet])
    # print(tc_matrix[romeo][romeo])
    # print(tc_matrix[love][love])
    # print(tc_matrix[juliet][love])
    # print(max(tc_matrix[0][1:]))
    # print(vocab_to_id)
    # for a in range(len(tc_matrix)):
    #     print(max(tc_matrix[a]))
    #     print(tc_matrix[a][a])
    #     print(tc_matrix[a].sort()[-2])

    return tc_matrix

def create_PPMI_matrix(term_context_matrix):
    """
    Given a term context matrix, output a PPMI matrix.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_context_matrix: A nxn numpy array, where n is
      the number of tokens in the vocab.

    Returns: A nxn numpy matrix, where A_ij is equal to the
       point-wise mutual information between the ith word
       and the jth word in the term_context_matrix.
    """

    n = term_context_matrix.shape[0]
    # words = 0
    # for i in range(n):
    #     words += term_context_matrix[i][i]
    words = np.trace(term_context_matrix)
    frequency_matrix = np.divide(term_context_matrix, words)

    # ppmi_matrix = np.zeros((n,n))

    diag = frequency_matrix.diagonal()
    div = np.matmul(diag, np.transpose(diag))

    prelog = np.divide(frequency_matrix, div)
    with np.errstate(divide='ignore'):
        ppmi_matrix = np.log(prelog)
    ppmi_matrix[np.isneginf(ppmi_matrix)]=0

    # print(term_context_matrix)

    # for i in range(n):
    #     for j in range(n):
    #         # print('ij')
    #         # print(term_context_matrix[i][j])
    #         # print('ii')
    #         # print(term_context_matrix[i][i])
    #         # print('jj')
    #         # print(term_context_matrix[j][j])
    #         if term_context_matrix[i][j] == 0:
    #             val = 0
    #         else:
    #
    #             val = math.log(frequency_matrix[i][j] / (frequency_matrix[i][i]*frequency_matrix[j][j]))
    #         ppmi_matrix[i][j] = val

    # print(ppmi_matrix)
    return ppmi_matrix

def create_tf_idf_matrix(term_document_matrix):
    """
    Given the term document matrix, output a tf-idf weighted version.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_document_matrix: Numpy array where each column represents a document
      and each row, the frequency of a word in that document.

    Returns:
      A numpy array with the same dimension as term_document_matrix, where
      A_ij is weighted by the inverse document frequency of document h.
    """
    # n = # documents
    n = term_document_matrix.shape[1]
    idf = np.sum(term_document_matrix, axis=1)
    idf = np.log(np.reciprocal(idf.astype(np.float64))*n)
    tf_idf_matrix = np.multiply(term_document_matrix.T, idf).T
    return tf_idf_matrix

def compute_cosine_similarity(vector1, vector2):
    """
    Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    """
    dp = np.dot(vector1, vector2)
    mag1 = np.sqrt(vector1.dot(vector1))
    mag2 = np.sqrt(vector2.dot(vector2))
    similarity = dp/(mag1*mag2)
    return similarity

def compute_jaccard_similarity(vector1, vector2):
    """
    Computes the jaccard similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    """
    dp = np.dot(vector1, vector2)
    c1 = vector1.dot(vector1)
    c2 = vector2.dot(vector2)
    similarity = dp / (c1 + c2 - dp)
    return similarity

def compute_dice_similarity(vector1, vector2):
    """
    Computes the dice similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    """
    dp = np.dot(vector1, vector2)
    c1 = vector1.dot(vector1)
    c2 = vector2.dot(vector2)
    similarity = 2*dp / (c1 + c2)
    return similarity

def rank_plays(target_play_index, term_document_matrix, similarity_fn):
    """
    Ranks the similarity of all of the plays to the target play.

    Inputs:
      target_play_index: The integer index of the play we want to compare all others against.
      term_document_matrix: The term-document matrix as a mxn numpy array.
      similarity_fn: Function that should be used to compared vectors for two
        documents. Either compute_dice_similarity, compute_jaccard_similarity, or
        compute_cosine_similarity.

    Returns:
      A length-n list of integer indices corresponding to play names,
      ordered by decreasing similarity to the play indexed by target_play_index
    """
    similarities = dict()

    target_play_vector = term_document_matrix[:,target_play_index]

    for i in range(term_document_matrix.shape[1]):
        if i != target_play_index:
            comparison_play_vector = term_document_matrix[:,i]
            similarity = similarity_fn(target_play_vector, comparison_play_vector)
            similarities[i] = similarity

    # print(similarities)
    # print(sorted(similarities, key=similarities.get))
    return sorted(similarities, key=similarities.get, reverse=True)

def rank_words(target_word_index, matrix, similarity_fn):
    """
    Ranks the similarity of all of the words to the target word.

    Inputs:
      target_word_index: The index of the word we want to compare all others against.
      matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
      similarity_fn: Function that should be used to compared vectors for two word
        ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
        compute_cosine_similarity.

    Returns:
      A length-n list of integer word indices, ordered by decreasing similarity to the
      target word indexed by word_index
    """
    similarities = dict()
    # print('matrix')
    # print(matrix)
    # print('target index is: ' + str(target_word_index))
    target_word_vector = matrix[target_word_index,:]

    for i in range(matrix.shape[0]):
        if i != target_word_index:
            comparison_word_vector = matrix[i,:]
            similarity = similarity_fn(target_word_vector, comparison_word_vector)
            similarities[i] = similarity

    # print(similarities)
    return sorted(similarities, key=similarities.get, reverse=True)


if __name__ == '__main__':
    tuples, document_names, vocab = read_in_shakespeare()

    print('Computing term document matrix...')
    td_matrix = create_term_document_matrix(tuples, document_names, vocab)

    print('Computing tf-idf matrix...')
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)

    print("Computing term context matrix...")
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

    print('Computing PPMI matrix...')
    PPMI_matrix = create_PPMI_matrix(tc_matrix)
    print(PPMI_matrix.dtype)

    random_idx = random.randint(0, len(document_names)-1)
    similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]

    for sim_fn in similarity_fns:
        print('\nThe top most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
        ranks = rank_plays(random_idx, td_matrix, sim_fn)
        for idx in range(0, 1):
            doc_id = ranks[idx]
            print('%d: %s' % (idx+1, document_names[doc_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on term-document frequency matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], td_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on tf idf matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], tf_idf_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))
