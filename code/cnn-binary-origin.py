# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import pickle
import sys

device = torch.device(0 if torch.cuda.is_available() else "cpu")
from copy import deepcopy
import random


# In[2]:

class CrossValidator:
    def __init__(self, model, data, compute_acc, loss_function,
                 partition=5, decoder=None, batch_size=100, epochs=20, lr=1e-3):
        self.model = model
        self.data = data
        self.data_size = len(data)
        self.partition = partition
        self.decoder = decoder
        self.compute_acc = compute_acc
        self.train_data = []
        self.val_X = []
        self.val_Y = []
        self.precision = 0
        self.recall = 0
        self.loss_history = []
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.precision_his = []
        self.recall_his = []

    def create_data(self, augment=True):
        prec = [int(i * self.data_size) for i in [0, 0.8, 0.9, 1]]
        self.data_X = train_X_new
        self.data_Y = train_Y_new
        train_X = [np.array(line) for line in self.data_X[prec[0]: prec[1]]]
        val_X = [np.array(line) for line in self.data_X[prec[1]: prec[2]]]
        test_X = [np.array(line) for line in self.data_X[prec[2]: prec[3]]]
        train_Y = [np.array(line) for line in self.data_Y[prec[0]: prec[1]]]
        val_Y = [np.array(line) for line in self.data_Y[prec[1]: prec[2]]]
        test_Y = [np.array(line) for line in self.data_Y[prec[2]: prec[3]]]

        train_X_augment = []
        train_Y_augment = []
        for i, target in enumerate(train_Y):
            train_X_augment.append(train_X[i])
            train_Y_augment.append(train_Y[i])
            for direction in [1]:
                for shift in range(1, 12):
                    for length in [0, 0.5]:
                        for silence_length in [0, 0.3]:
                            train_X_temp = deepcopy(train_X[i])
                            train_X_temp[1, :] += direction * shift
                            train_X_temp[2, :] += silence_length
                            train_X_temp[0, :] += length
                            train_X_augment.append(train_X_temp)
                            train_Y_augment.append(train_Y[i])
        train_data = []
        if augment == True:
            for i, x in enumerate(train_Y_augment):
                train_data.append([train_X_augment[i], train_Y_augment[i]])
        else:
            for i, x in enumerate(train_Y):
                train_data.append([train_X[i], train_Y[i]])
        return train_data, val_X, val_Y, test_X, test_Y

    def tensorize(self, p):
        p = np.array(p)
        p = torch.from_numpy(p).float()
        p = p.to(device)
        return p

    def create_batch(self, index):
        output_X = []
        output_Y = []
        for i in range(self.batch_size):
            if self.batch_size * index + i < len(self.train_data):
                output_X.append(self.train_data[self.batch_size * index + i][0])
                output_Y.append(self.train_data[self.batch_size * index + i][1])
        return output_X, output_Y

    def compute(self):
        self.train_data, self.val_X, self.val_Y, self.test_X, self.test_Y = self.create_data(augment=True)
        self.val_X = self.tensorize(self.val_X)
        self.val_Y = self.tensorize(self.val_Y)
        self.val_Y = self.val_Y.long()
        self.test_X = self.tensorize(self.test_X)
        self.test_Y = self.tensorize(self.test_Y).long()
        cur_model = deepcopy(self.model).to(device)
        optimizer = optim.Adam(cur_model.parameters(), lr=self.lr, weight_decay=5e-8)
        loss = 0
        train_len = len(self.train_data)
        self.loss_history.append([])
        for j in range(self.epochs):
            random.shuffle(self.train_data)
            for k in range(int(train_len / self.batch_size)):
                batch_X, batch_Y = self.create_batch(k)
                batch_X = self.tensorize(batch_X)
                batch_Y = self.tensorize(batch_Y)
                batch_Y = batch_Y.long()
                optimizer.zero_grad()
                output_Y = cur_model(batch_X)
                loss = self.loss_function(output_Y, batch_Y)
                loss.backward()
                # print(loss)
                optimizer.step()
            # self.loss_history[i].append(loss.cpu().data)
                if k % 100 == 0:
                    output_Y = cur_model(self.val_X)
                    if self.decoder != None: output_Y = self.decoder(output_Y)
                    temp_precision, temp_recall = self.compute_acc(output_Y, self.val_Y)
                    self.precision_his.append(temp_precision)
                    self.recall_his.append(temp_recall)
                    print(k, temp_precision, temp_recall)
                    sys.stdout.flush()

        output_Y = cur_model(self.test_X)
        if self.decoder != None: output_Y = self.decoder(output_Y)
        temp_precision, temp_recall = self.compute_acc(output_Y, self.test_Y)
        self.precision_his.append(temp_precision)
        self.recall_his.append(temp_recall)
        print(temp_precision, temp_recall, 2 * temp_precision * temp_recall / ((temp_precision + temp_recall) + 0.0001))
        sys.stdout.flush()

        return temp_precision, temp_recall, self.loss_history


# In[3]:

class CnnMusic(nn.Module):
    def __init__(self):
        super(CnnMusic, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, 3, padding=2)
        self.conv2 = nn.Conv1d(32, 128, 3, padding=2)
        self.conv3 = nn.Conv1d(128, 64, 3, padding=2)
        self.conv4 = nn.Conv1d(64, 64, 3, padding=2)
        self.conv5 = nn.Conv1d(64, 64, 3, padding=2)
        self.upsample1 = nn.ConvTranspose1d(64, 64, 3, padding=1, output_padding=1, stride=2)
        self.conv6 = nn.Conv1d(128, 64, 3, padding=2)
        self.upsample2 = nn.ConvTranspose1d(64, 64, 3, padding=1, output_padding=1, stride=2)
        self.conv7 = nn.Conv1d(128, 128, 3, padding=2)
        self.upsample3 = nn.ConvTranspose1d(128, 128, 3, padding=1, output_padding=1, stride=2)
        self.conv8 = nn.Conv1d(256, 128, 3, padding=2)
        self.upsample4 = nn.ConvTranspose1d(128, 128, 3, padding=1, output_padding=1, stride=2)
        self.conv9 = nn.Conv1d(160, 40, 3, padding=2)
        self.conv10 = nn.Conv1d(40, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = x[:, :, :-2]
        skip1 = x
        x = F.max_pool1d(x, 2)
        x = self.conv2(x)
        x = x[:, :, :-2]
        skip2 = x
        x = F.max_pool1d(x, 2)
        x = self.conv3(x)
        x = x[:, :, :-2]
        skip3 = x
        x = F.max_pool1d(x, 2)
        x = self.conv4(x)
        x = x[:, :, :-2]
        skip4 = x
        x = F.max_pool1d(x, 2)
        x = self.conv5(x)
        x = x[:, :, :-2]
        x = self.upsample1(x)
        x = torch.cat((x, skip4), dim=1)
        x = self.conv6(x)
        x = x[:, :, :-2]
        x = self.upsample2(x)
        # x=self.upsample2(self.conv6(x))
        x = torch.cat((x, skip3), dim=1)
        x = self.conv7(x)
        x = x[:, :, :-2]
        x = self.upsample3(x)
        # x=self.upsample3(self.conv7(x))
        x = torch.cat((x, skip2), dim=1)
        x = self.conv8(x)
        x = x[:, :, :-2]
        x = self.upsample4(x)
        # x=self.upsample4(self.conv8(x))
        x = torch.cat((x, skip1), dim=1)
        x = self.conv9(x)
        x = x[:, :, :-2]
        x = self.conv10(x)
        return x


# In[4]:

model = CnnMusic()


def decode(p):
    p = torch.argmax(p, dim=1)
    return p


def compute_acc(prediction, ground_truth):
    # print("into func")
    prediction = prediction.cpu()
    ground_truth = ground_truth.cpu()
    b, l = prediction.shape
    cor = 0
    tot1 = 0
    tot2 = 0
    for i in range(b):
        for j in range(l):
            if ground_truth[i][j] == 1: tot1 += 1
            if prediction[i][j] == 1: tot2 += 1
            if prediction[i][j] == 1 and ground_truth[i][j] == 1: cor += 1
    # print("out func")
    return cor / max(1, tot2), cor / max(1, tot1)


# In[5]:

with open("/gpfsnyu/home/yz6492/melody_seg/data/essen_binary.pkl", "rb") as f:
    dic = pickle.load(f)
    data_X = dic["X"]
    data_Y = dic["Y"]

train_X_new = []
train_Y_new = []

for i, x in enumerate(data_X):
    temp_x = np.zeros((3, 400))
    temp_y = np.zeros((400))
    l = len(data_X[i])
    for j in range(400):
        temp_x[0][j] = data_X[i][j % l][1] - data_X[i][j % l][0]
        temp_x[1][j] = data_X[i][j % l][2]
        if (j % l) != l - 1:
            temp_x[2][j] = data_X[i][(j + 1) % l][0] - data_X[i][j % l][1]
        else:
            temp_x[2][j] = random.random()
        temp_y[j] = data_Y[i][j % l]
    train_X_new.append(temp_x)
    train_Y_new.append(temp_y)

data = []
for i, x in enumerate(train_X_new):
    data.append([train_X_new[i], train_Y_new[i]])

print(len(data))
# In[6]:
w = torch.Tensor([1.0, 2.0]).to(device)
validator = CrossValidator(model, data, compute_acc=compute_acc, loss_function=nn.CrossEntropyLoss(weight=w),
                           decoder=decode, lr=1e-3, epochs=10)
precision, recall, loss_history = validator.compute()

# In[7]:

import pickle

f = open('unet_val_crossentropy_augmented.pk', 'wb')
pickle.dump(precision, f)
pickle.dump(recall, f)
pickle.dump(loss_history, f)
f.close()
