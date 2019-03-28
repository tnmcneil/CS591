import torch
import glob
import unicodedata
import string

ALL_LETTERS = string.ascii_letters + ".,;'"
N_LETTERS = len(ALL_LETTERS)
ALL_FILENAMES_TRAIN = glob.glob('./cities_train/train/*.txt')
ALL_FILENAMES_VAL = glob.glob('./cities_val/val/*.txt')

def unicode_to_ascii(str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', str)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


def readlines(filename):
    filename = filename.replace('\\', '/')
    lines = open(filename, encoding="utf8", errors='ignore').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def prep_data():
    category_lines_train = {}
    category_lines_val = {}
    all_categories = []
    for filename in ALL_FILENAMES_TRAIN:
        category = filename.split('/')[-1].split('\\')[-1].split('.')[0]
        all_categories.append(category)
        lines = readlines(filename)
        category_lines_train[category] = lines

    for filename in ALL_FILENAMES_VAL:
        category = filename.split('/')[-1].split('\\')[-1].split('.')[0]
        lines = readlines(filename)
        category_lines_val[category] = lines

    n_categories = len(all_categories)
    return n_categories, category_lines_train, category_lines_val, all_categories


# turn a letter into a <1 x n_letters> tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    letter_index = ALL_LETTERS.find(letter)
    tensor[0][letter_index] = 1
    return tensor


# turn a line into a <line_length x 1 x n_letters>
# or an array of one hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        letter_index = ALL_LETTERS.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor


n_categories, category_lines_train, category_lines_val, all_categories = prep_data()


# def main():
#     n_categories, category_lines_train, category_lines_val, all_categories = prep_data()
#     # print(unicode_to_ascii('Ślusàrski'))
#     # print('n categories =', n_categories)
#     # print(category_lines['Italian'][:5])
#     # print(letter_to_tensor('J'))
#     # print(line_to_tensor('Jones').size())
#
#
# if __name__ == '__main__':
#     main()