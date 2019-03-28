import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from data import *
from model import *
import random
import time
import math


N_HIDDEN = 128
N_EPOCHS = 100000
PRINT_EVERY = 5000
PLOT_EVERY = 1000
LEARNING_RATE = 0.002


def category_from_output(output):
    top_n, top_i = output.data.topk(1)      # tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


def random_training_pair():
    category = all_categories[random.randint(0, len(all_categories) - 1)]
    line = category_lines_train[category][random.randint(0, len(category_lines_train[category])-1)]
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor


rnn = RNN(N_LETTERS, N_HIDDEN, n_categories)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=LEARNING_RATE)


def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.data


# keep track of losses for plotting
all_losses = []
current_loss = 0


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for epoch in range(1, N_EPOCHS +1):
    category, line, category_tensor, line_tensor = random_training_pair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if epoch % PRINT_EVERY == 0:
        guess, guess_i = category_from_output(output)
        correct = '!' if guess == category else 'X (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' %
              (epoch, epoch / N_EPOCHS * 100, time_since(start), loss, line, guess, correct))

    if epoch % PLOT_EVERY == 0:
        all_losses.append(current_loss / PLOT_EVERY)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')

plt.figure()
plt.plot(all_losses)
plt.show()


confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000
def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


# create confusion matrix
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_training_pair()
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# normalize rows by dividing by sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

