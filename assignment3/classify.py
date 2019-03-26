import numpy as np
import torch
import glob
import unicodedata
import string
import torch.nn as nn
from torch.autograd import Variable

# Goal: char-RNN that classifies baby names by country of origin
all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)


def unicode_to_ascii(str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', str)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readlines(filename):
    filename = filename.replace('\\', '/')
    lines = open(filename, encoding="utf8").read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def prep_data():
    all_filenames = glob.glob('./data/names/*.txt')
    category_lines = {}
    all_categories = []
    for filename in all_filenames:
        category = filename.split('/')[-1].split('\\')[-1].split('.')[0]
        all_categories.append(category)
        lines = readlines(filename)
        category_lines[category] = lines
    n_categories = len(all_categories)
    return n_categories, category_lines, all_categories


# turn a letter into a <1 x n_letters> tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor


# turn a line into a <line_length x 1 x n_letters>
# or an array of one hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def main():
    n_categories, category_lines, all_categories = prep_data()
    print('number of categories =  %d' % (n_categories))
    print(n_letters)


if __name__ == '__main__':
    main()