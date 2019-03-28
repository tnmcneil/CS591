from model import *
from data import *
import sys

rnn = torch.load('char-rnn-classification.pt')


# return output prediction given single line
def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(line, n_predictions=3):
    output = evaluate(Variable(line_to_tensor(line)))
    # get top n category predictions
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []
    for i in range(n_predictions):
        val = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (val, all_categories[category_index]))
        predictions.append([val, all_categories[category_index]])

    return predictions


def main():
    line = sys.argv[1]
    predict(line)


if __name__ == '__main__':
    main()