from __future__ import unicode_literals, print_function, division
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!pip install wandb
import wandb
wandb.login(key = 'da5365b4335ad8c7a1df7f3653ec9d0b092e8b09')


from io import open
import unicodedata
import re
import random
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
# import tensorflow as tf
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("TPU" if tf.config.list_physical_devices('GPU') else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in list(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs_train(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    data = pd.read_csv('/kaggle/input/aksharantar-sampled/aksharantar_sampled/hin/hin_train.csv')
    n_data = np.array(data)
    l_data = [list(n_data[i]) for i in range(len(n_data))]

    # Split every line into pairs
    pairs = l_data
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def readLangs_valid(lang1, lang2, reverse=False):
    data = pd.read_csv('/kaggle/input/aksharantar-sampled/aksharantar_sampled/hin/hin_valid.csv')


    n_data = np.array(data)
    l_data = [list(n_data[i]) for i in range(len(n_data))]

    # Split every line into pairs
    pairs = l_data
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def readLangs_test(lang1, lang2, reverse=False):
    data = pd.read_csv('/kaggle/input/aksharantar-sampled/aksharantar_sampled/hin/hin_test.csv')


    n_data = np.array(data)
    l_data = [list(n_data[i]) for i in range(len(n_data))]

    # Split every line into pairs
    pairs = l_data
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData_train(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs_train(lang1, lang2, reverse)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

def prepareData_valid(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs_valid(lang1, lang2, reverse)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs


def prepareData_test(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs_test(lang1, lang2, reverse)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData_test('eng', 'hin')
print(random.choice(pairs))

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in list(sentence)]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader_train(batch_size):
    input_lang, output_lang, pairs = prepareData_train('eng', 'hin')

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

def get_dataloader_valid(batch_size):
    input_lang, output_lang, pairs = prepareData_valid('eng', 'hin')

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    valid_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)
    return input_lang, output_lang, valid_dataloader

def get_dataloader_test(batch_size):
    input_lang, output_lang, pairs = prepareData_test('eng', 'hin')

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    test_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    test_sampler = RandomSampler(test_data)
    test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return input_lang, output_lang, test_dataloader

MAX_LENGTH = 67
batch_size = 1024


input_lang, output_lang, train_dataloader = get_dataloader_train(batch_size)
_,_, valid_dataloader = get_dataloader_valid(batch_size)
_,_, test_dataloader = get_dataloader_test(batch_size)

def get_cell(cell_type: str):
    if cell_type == "LSTM":
        return nn.LSTM
    elif cell_type == "GRU":
        return nn.GRU
    elif cell_type == "RNN":
        return nn.RNN
    else:
        raise Exception("Invalid cell type")

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout, cell_type):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout)




    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout,cell_type):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.cell_type = cell_type
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout)


    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        # Compute attention
        query = hidden[-1].unsqueeze(1)  # Extract the last layer's hidden state
        context, _ = self.attention(query, encoder_outputs)

        # Concatenate embedded input and context
        input_with_context = torch.cat((embedded, context), dim=2)

        # Forward pass through the LSTM
        output, (hidden, cell) = self.rnn(input_with_context, (hidden, cell))

        # Output prediction
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs









def translate(model, src):
    inputs = [0]
    hidden, cell = model.encoder(src)
    for _ in range(67):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == 1:
                break
#     output_word = convert_text(inputs)
    return inputs

def model_accuracy(model,data_loader):
    acc =[]
    for i, batch in enumerate(data_loader):
        src = (batch[0]).to(device)
        trg = (batch[1]).to(device)
        for j in range(len(src)):
            src_j = src[j].reshape(67,1)
            trg_j = trg[j]

        # src = [src length, batch size]
        # trg = [trg length, batch size]
#         trg = trg[1:].reshape(-1)
            output_j = translate(model,src_j)
            if output_j[1:] == trg_j :
                acc.append(1)
            else:
                acc.append(0)
#             print(len(batch))
    accuracy = np.sum(acc)/len(acc)
#     print(acc)
    return accuracy

def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = (batch[0].T).to(device)
        trg = (batch[1].T).to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].reshape(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
#         print(output.shape,trg.shape)
#         print()
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = (batch[0].T).to(device)
            trg = (batch[1].T).to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].reshape(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
# model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(f"The model has {count_parameters(model):,} trainable parameters")

def train_model(model, train_dataloader,valid_dataloader,test_dataloader, optimizer, criterion, clip, teacher_forcing_ratio, device ,n_epochs):
    for epoch in range(1,n_epochs+1):
        train_loss = train_fn(
            model,
            train_dataloader,
            optimizer,
            criterion,
            clip,
            teacher_forcing_ratio,
            device,
        )
        valid_loss = evaluate_fn(
            model,
            valid_dataloader,
            criterion,
            device,
        )
        valid_acc = model_accuracy(model,valid_dataloader)
        if True:
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} |", end="")
            print(f"Val Loss: {valid_loss:.4f} |", end="")
            print(f"Val Accuracy: {valid_acc:.4%}")
            wandb.log({
                "epoch": epoch + 1,
                "training_loss": train_loss,
                "validation_accuracy": valid_acc,
                "validation_loss": valid_loss
            })
    test_acc = model_accuracy(model,test_dataloader)
    print(f"Test Accuracy: {test_acc:.4%}")
    wandb.log({"test_accuracy": test_acc})

n_epochs = 70
clip = 0
teacher_forcing_ratio = 0.75
criterion = nn.CrossEntropyLoss()
input_dim = input_lang.n_words
output_dim = output_lang.n_words
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
batch_size = 32
cell_type = 'LSTM'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_lang, output_lang, train_dataloader = get_dataloader_train(batch_size)
_,_, valid_dataloader = get_dataloader_valid(batch_size)
_,_, test_dataloader = get_dataloader_test(batch_size)
encoder = Encoder(input_dim, encoder_embedding_dim, hidden_dim, n_layers, encoder_dropout,cell_type)
decoder = Decoder(output_dim,decoder_embedding_dim,hidden_dim,n_layers,decoder_dropout,cell_type)
model = Seq2Seq(encoder, decoder, device).to(device)
optimizer = optim.Adam(model.parameters(),lr = .001)
train_model(model, train_dataloader,valid_dataloader,test_dataloader, optimizer, criterion, clip, teacher_forcing_ratio, device ,n_epochs)

sweep_configuration = {
    "method": "bayes",
    "metric": {
        "name": "validation_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "embed_size": {
            "values": [32,64,128]
        },
        "hidden_size": {
            "values": [128, 256, 512]
        },
        "cell_type": {
            "values": ["GRU", "LSTM", "RNN"]
        },
        "num_layers": {
            "values": [1, 2, 3]
        },
        "dropout": {
            "values": [0, 0.1, 0.2]
        },
        "learning_rate": {
            "values": [0.0005, 0.001, 0.005]
        },
        "optimizer": {
            "values": ["Sgd", "Adam","Nadam"]
        },
        "teacher_forcing_ratio": {
            "values": [0.5, 0.75, 0.25]
        }
    }
}

def train_sweep():

    run = wandb.init()
    config = wandb.config
    run.name = "embed_size {}_hidden_size {}_cell_type {}_num_layers {} _dropout {} _learning_rate {} _optimizer {} _teacher_forcing_ratio {}".format(config.embed_size, config.hidden_size, config.cell_type, config.num_layers, config.dropout, config.learning_rate, config.optimizer, config.teacher_forcing_ratio)


    batch_size = 128
    input_lang, output_lang, train_dataloader = get_dataloader_train(batch_size)
    _,_, valid_dataloader = get_dataloader_valid(batch_size)
    _,_, test_dataloader = get_dataloader_test(batch_size)
    n_epochs = 60
    clip = 0
    cell_type = config.cell_type
    teacher_forcing_ratio = config.teacher_forcing_ratio
    learning_rate = config.learning_rate
    optimiz = config.optimizer
    criterion = nn.CrossEntropyLoss()
    input_dim = input_lang.n_words
    output_dim = output_lang.n_words
    encoder_embedding_dim = config.embed_size
    decoder_embedding_dim = config.embed_size
    hidden_dim = config.hidden_size
    n_layers = config.num_layers
    encoder_dropout = config.dropout
    decoder_dropout = config.dropout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LENGTH = 67
    encoder = Encoder(input_dim,encoder_embedding_dim,hidden_dim,n_layers,encoder_dropout,cell_type)
    decoder = Decoder(output_dim,decoder_embedding_dim,hidden_dim,n_layers,decoder_dropout,cell_type)
    model = Seq2Seq(encoder, decoder, device).to(device)
    print(model.apply(init_weights))
    print(f"The model has {count_parameters(model):,} trainable parameters")
    if optimiz == "Adam":
        optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    elif optimiz == "NAdam":
        optimizer = optim.NAdam(model.parameters(),lr = learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(),lr = learning_rate)
    train_model(model, train_dataloader,valid_dataloader,test_dataloader, optimizer, criterion, clip, teacher_forcing_ratio, device,n_epochs)
    run.finish()

wandb_id = wandb.sweep(sweep_configuration, project="CS6910_Assn3_AttnRNN")
wandb.agent(wandb_id, train_sweep)

import csv
import random
import pandas as pd
import matplotlib.pyplot as plt

def generate_predictions(model, data_loader, device):
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for batch in data_loader:
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output.argmax(dim=-1)  # get predicted token indices
            predictions.extend(output.cpu().numpy().tolist())
            ground_truths.extend(trg[1:].cpu().numpy().tolist())  # exclude SOS token
    return predictions, ground_truths

def save_to_csv(inputs, predictions, ground_truths, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Input', 'Prediction', 'Ground Truth'])
        for input_seq, prediction_seq, ground_truth_seq in zip(inputs, predictions, ground_truths):
            input_text = ' '.join([input_lang.index2word[idx] for idx in input_seq])
            prediction_text = ' '.join([output_lang.index2word[idx] for idx in prediction_seq])
            ground_truth_text = ' '.join([output_lang.index2word[idx] for idx in ground_truth_seq])
            writer.writerow([input_text, prediction_text, ground_truth_text])

def select_random_samples(data_loader, num_samples):
    samples = random.sample(data_loader.dataset, num_samples)
    return [src for src, _ in samples], [trg for _, trg in samples]


# Generate random samples
random_inputs, random_ground_truths = select_random_samples(test_dataloader, 15)

# Generate predictions for random samples
random_predictions, _ = generate_predictions(model, random_inputs, device)

# Save predictions to CSV
save_to_csv(random_inputs, random_predictions, random_ground_truths, '/kaggle/working/random_15_predictions.csv')



# Read the CSV file
df = pd.read_csv('/kaggle/working/random_15_predictions.csv')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create a table from the DataFrame
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# Adjust the table properties
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

# Remove axes and ticks
ax.axis('off')

# Set the plot title
plt.title('Random 15 Predictions')

# Save the plot
plt.savefig('/kaggle/working/random_15_predictions_table.png')

# Show the plot
plt.show()


from PIL import Image
# Initialize WandB with your project name
wandb.init(project="CS6910_Assignment3_Attention")
# List of tuples containing paths to your images and their respective names
image_info = [
    ("/kaggle/working/random_15_predictions_table.png", "predictions on test set with attention")
]

# Log each image with its name
for i, (path, name) in enumerate(image_info):
    image = Image.open(path)
    wandb.log({f"{name}": wandb.Image(image)}, commit=False)

# Finish the logging and create a grid visualization
wandb.finish()


def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions[0, :len(output_words), :])

# Initialize wandb
wandb.init(project="CS6910_Assignment3_Attention")

# Function to generate random plots with input word names
def generate_random_plots(model, data_loader, device, num_plots=9):
    # Get random indices
    random_indices = random.sample(range(len(data_loader.dataset)), num_plots)
    
    # Iterate over random indices
    for i, idx in enumerate(random_indices):
        # Show attention for each random example
        show_attention(model, data_loader, device, idx)
        
        # Get input sentence
        input_sentence = [input_lang.index2word[idx.item()] for idx in data_loader.dataset[idx][0]]
        
        # Set plot title and save
        plt.title(' '.join(input_sentence))
        image_path = f'/kaggle/working/plot_{i}.png'
        plt.savefig(image_path)
        plt.close()
        
        # Log the image with its name
        wandb.log({f'{i}': wandb.Image(image_path, caption=' '.join(input_sentence))})

# Generate random plots
generate_random_plots(model, test_dataloader, device)


