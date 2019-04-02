import time
import math
import torch
from torch.autograd import Variable
import argparse
import os

from data import *
from model import *
from generate import *


# parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=50)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
args = argparser.parse_args()

file, file_len = read_file(args.filename)


# assemble pair of input and target tensors for training from random chunk
# input = all characters up to last
# target = all characters from the first
# eg chunk='abc' --> input='ab', target='bc'
def random_training_set():
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m*60
    return '%dm %ds' % (m,s)


decoder = RNN(n_characters, args.hidden_size, n_characters, args.n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0


def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].unsqueeze(0))

    loss.backward()
    decoder_optimizer.step()

    return loss.data.item() / args.chunk_len


def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


try:
    print('Training for %d epochs ...' % args.n_epochs)
    for epoch in range(1, args.n_epochs + 1):
        loss = train(*random_training_set())
        loss_avg += loss

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            print(generate(decoder, 'Wh', 100), '\n')

    print("Saving model...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()