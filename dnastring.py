"""
Learning how to use Neural Networks with genomic data
For now positive examples are strings of 'A' and negative examples are strings of 'C'
"""
import codecs
import errno
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from torch import nn
from torch import optim
import torch.utils.data as data
import os.path
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Parameters
alphabetSize = 4
num_epochs = 50
n_hidden = 64
n_categories = 2
learning_rate = 0.005

dlParams = {'batch_size': 5,
            'shuffle': True,
            'num_workers': 6}
      
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

baseToInt = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'N': 4
}

# Returns a vector of zeroes for padding
def zeroElement():
    return [0 for i in range(alphabetSize)]

# Converts a DNA string to a vector with one-hot encoding
def dnaToVec(s):
    n = len(s)
    res = [[0 for i in range(alphabetSize)] for i in range(n)]
    for i in range(n):
        res[i][baseToInt[s[i]]] = 1
    return res

# Converts a set of labelled DNA strings into a PyTorch dataset
class DNAStringData(data.Dataset):
    def __init__(self, samples, labels, maxlen, train = True):
        self.maxlen = maxlen
        self.train = train
        if self.train:
            self.training_data = []
            self.training_labels = []
            self.lengths = []
            for i in range(0, len(samples)):
                padded = self.pad(dnaToVec(samples[i]))
                self.training_data.append(padded)
                self.training_labels.append(labels[i])
                self.lengths.append(len(samples[i]))
            self.training_data = torch.FloatTensor(self.training_data)
        else:
            self.testing_data = []
            self.testing_labels = []
            self.lengths = []
            for i in range(0, len(samples)):
                padded = self.pad(dnaToVec(samples[i]))
                self.testing_data.append(padded)
                self.testing_labels.append(labels[i])
                self.lengths.append(len(samples[i]))
            self.testing_data = torch.FloatTensor(self.testing_data)
            
    def __len__(self):
        if self.train:
            return len(self.training_data)
        else:
            return len(self.testing_data)
            
    def __getitem__(self, index):
        if self.train:
            return self.training_data[index], self.training_labels[index], self.lengths[index]
        else:
            return self.testing_data[index], self.testing_labels[index], self.lengths[index]
       
    # Pads data with zeroes     
    def pad(self, s):
        padded = [zeroElement() for i in range(self.maxlen)] 
        if len(s) > self.maxlen:
            for i in range(self.maxlen):
                padded[i] = s[i]
        else:
            for i in range(len(s)):
                padded[i] = s[i]
        return padded
    
def genToyData(train = True):
    if train:
        training_data = []
        training_labels = []
        # Class 0 is strings of A's, Class 1 is strings of C's
        for i in range(1, 11):
            training_data.append('A' * i)
            training_labels.append(1)
            training_data.append('C' * i)
            training_labels.append(0)
        return training_data, training_labels
    else:
        testing_data = []
        testing_labels = []
        # All of these data points are similar to training data so should be predicted correctly
        for i in range(11, 21):
            testing_data.append('A' * i)
            testing_labels.append(1)
            testing_data.append('C' * i)
            testing_labels.append(0)
        # Add data point that should be predicted incorrectly based on training data
        testing_data.append('AAA')
        testing_labels.append(0)
        return testing_data, testing_labels

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Trains the network on a particular data point
def train(rnn, category_tensor, line_tensor, length, criterion):
    hidden = rnn.initHidden()

    rnn.zero_grad()
    for i in range(length):
        output, hidden = rnn(torch.unsqueeze(line_tensor[i], 0), hidden)
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

# Tests the network on a particular data point  
def test(rnn, category_tensor, line_tensor, length):
    hidden = rnn.initHidden()
    for i in range(length):
        output, hidden = rnn(torch.unsqueeze(line_tensor[i], 0), hidden)
    return output

def runNN(xtrain, ytrain, xtest, ytest):
    
    # Generate DataLoaders for each of the datasets
    maxlen = max(len(max(xtrain, key = len)), len(max(xtest, key = len)))
    trainDataset = DNAStringData(xtrain, ytrain, maxlen, train = True)
    trainDL = data.DataLoader(dataset = trainDataset, **dlParams)
    testDataset = DNAStringData(xtest, ytest, maxlen, train = False)
    testDL = data.DataLoader(dataset = testDataset, **dlParams)
    
    # Initialize RNN and loss function
    net = RNN(alphabetSize, n_hidden, n_categories)
    criterion = nn.NLLLoss()
    
    # Run on training data for a fixed number of epochs
    for epoch in range(num_epochs):
        totalLoss = 0
        for batch_idx, (x, y, length) in enumerate(trainDL):
            for i in range(len(x)):
                xx = x[i].to(device)
                _, curLoss = train(net, torch.LongTensor([y[i]]), xx, length[i], criterion)
                totalLoss += curLoss
        if epoch%10 == 9:
            print('Epoch: ' + str(epoch+1) + '; Loss: ' + str(totalLoss))
    
    # Run on test data and assess accuracy
    correct = 0
    incorrect = 0
    for batch_idx, (x, y, length) in enumerate(testDL):
        for i in range(len(x)):
            xx = x[i].to(device)
            out = test(net, y, x[i], length[i])
            predictedClass = torch.argmax(out)
            print(predictedClass)
            if predictedClass == y[i]:
                correct += 1
            else:
                incorrect += 1
    print('Correct: ' + str(correct) + '; Incorrect: ' + str(incorrect)) 

def main():
    xtrain, ytrain = genToyData(True)
    xtest, ytest = genToyData(False)
    runNN(xtrain, ytrain, xtest, ytest)
    
    
if __name__ == "__main__": main()

