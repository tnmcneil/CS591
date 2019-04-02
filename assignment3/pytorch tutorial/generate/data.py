import unidecode
import string
import random
import time
import math
import torch
from torch.autograd import Variable

all_characters = string.printable
n_characters = len(all_characters)

file = unidecode.unidecode(open('./shakespeare.txt').read())
file_len = len(file)

CHUNK = 200


def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


def random_chunk():
    start = random.randint(0, file_len - CHUNK)
    end = start + CHUNK + 1
    return file[start:end]


# string --> long tensor by looping through string and looking up index of each char
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)