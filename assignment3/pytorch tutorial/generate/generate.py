import torch
import argparse

from data import *
from model import *


# feed one character at a time, use outputs of network as pdf for next char, repeat
# to start generation pass priming string to build from hidden state
def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # use priming string to build up hidden state
    for p in range(len(prime_str)-1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # sample from network as multinomial dist
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    del args.filename
    print(generate(decoder, **vars(args)))


if __name__ == '__main__':
    main()