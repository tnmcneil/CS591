import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math

from data import *
from model import *


N_HIDDEN = 192
N_EPOCHS = 100000
PRINT_EVERY = 5000
PLOT_EVERY = 1000
LEARNING_RATE = 0.0005


def category_from_output(output):
    top_n, top_i = output.data.topk(1)      # tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


def random_pair(category_lines):
    category = all_categories[random.randint(0, len(all_categories) - 1)]
    line = category_lines[category][random.randint(0, len(category_lines[category]) - 1)]
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor


# def random_training_pair():
#     category = all_categories[random.randint(0, len(all_categories) - 1)]
#     line = category_lines_train[category][random.randint(0, len(category_lines_train[category])-1)]
#     category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
#     line_tensor = Variable(line_to_tensor(line))
#     return category, line, category_tensor, line_tensor


rnn = RNN(N_LETTERS, N_HIDDEN, n_categories)
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.SGD(rnn.parameters(), lr=LEARNING_RATE)


def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.data


def validate(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.data


# keep track of losses for plotting
train_losses = []
val_losses = []
current_train_loss = 0
current_val_loss = 0


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for epoch in range(1, N_EPOCHS + 1):
    # training
    category_train, line_train, category_tensor_train, line_tensor_train = random_pair(category_lines_train)
    train_output, train_loss = train(category_tensor_train, line_tensor_train)
    current_train_loss += train_loss
    guess, guess_i = category_from_output(train_output)


    if epoch % PRINT_EVERY == 0:
        correct = '!' if guess == category_train else 'X (%s)' % category_train
        print('%d %d%% (%s) %.4f %s / %s %s' %
              (epoch, epoch / N_EPOCHS * 100, time_since(start), train_loss, line_train, guess, correct))


    if epoch % PLOT_EVERY == 0:
        train_losses.append(current_train_loss / PLOT_EVERY)
        current_train_loss = 0

    # validating
    category_val, line_val, category_tensor_val, line_tensor_val = random_pair(category_lines_val)
    val_output, val_loss = validate(category_tensor_val, line_tensor_val)
    current_val_loss += val_loss

    if epoch % PLOT_EVERY == 0:
        val_losses.append(current_val_loss / PLOT_EVERY)
        current_val_loss = 0



    # for category in category_lines_train:
    #     print('training category %s' % category)
    #     for line in category_lines_train[category]:
    #         category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    #         line_tensor = Variable(line_to_tensor(line))
    #
    #         train_output, train_loss = train(category_tensor, line_tensor)
    #         current_train_loss += train_loss
    #
    #         if epoch % PRINT_EVERY == 0:
    #             guess, guess_i = category_from_output(train_output)
    #             correct = '!' if guess == category else 'X (%s)' % category
    #             print('%d %d%% (%s) %.4f %s / %s %s' %
    #                   (epoch, epoch / N_EPOCHS * 100, time_since(start), train_loss, line, guess, correct))
    #
    #         if epoch % PLOT_EVERY == 0:
    #             train_losses.append(current_train_loss / PLOT_EVERY)
    #             current_train_loss = 0
    # for category in category_lines_val:
    #     print('validating category %s' % category)
    #     for line in category_lines_val[category]:
    #         category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    #         line_tensor = Variable(line_to_tensor(line))
    #
    #         val_output, val_loss = validate(category_tensor, line_tensor)
    #         current_val_loss += val_loss
    #
    #         if epoch % PLOT_EVERY == 0:
    #             val_losses.append(current_val_loss / PLOT_EVERY)
    #             current_val_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')

plt.figure()
plt.plot(train_losses, label='training')
plt.plot(val_losses, label='validation')
plt.legend(loc='upper right')
plt.xlabel('Number of Batches (in 1000s)')
plt.ylabel('Loss')
plt.show()


confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000
def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


val_accuracies_macro = [0]*9
val_accuracy_micro = 0
val_count = 0


# create confusion matrix
# for i in range(n_confusion):
#     category, line, category_tensor, line_tensor = random_pair(category_lines_val)  # random_training_pair()
#     output = evaluate(line_tensor)
#     guess, guess_i = category_from_output(output)
#     category_i = all_categories.index(category)
#     confusion[category_i][guess_i] += 1
for category in category_lines_val:
    for line in category_lines_val[category]:
        category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
        line_tensor = Variable(line_to_tensor(line))
        output = evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
        val_count += 1
        if category_i == guess_i:
            val_accuracy_micro += 1
            val_accuracies_macro[category_i] += 1


print(val_count)

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

val_accuracy_micro = val_accuracy_micro / val_count
val_accuracies_macro = [i / 100 for i in val_accuracies_macro]
val_accuracies_macro = np.asarray(val_accuracies_macro)
macro = np.mean(val_accuracies_macro)

print('micro accuracy is %f' % val_accuracy_micro)
print('macro accuracy is %f' % macro)



