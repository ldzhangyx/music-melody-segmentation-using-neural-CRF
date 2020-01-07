
# coding: utf-8

# In[1]:



# coding: utf-8

# naive LSTM model trained with smooth data

import torch
import torch.nn as nn
from torch import optim
import sys
import torch.nn.functional as F
import pickle
import math
import time
import numpy as np
import copy
from torch.autograd import Variable

device = torch.device(0 if torch.cuda.is_available() else "cpu")
    
class DecoderRNN(nn.Module):
    def __init__(self, input_size, augmented_size, hidden_size, output_size, dropout_p = 0):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.augmented_size = augmented_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.verbose = (self.dropout_p != 0)

        self.lstm_1 = nn.LSTM(self.input_size,  self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_2 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_3 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_4 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_5 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_6 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.lstm_7 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers=1, bidirectional=True)
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.dropout4 = nn.Dropout(self.dropout_p)
        self.dropout5 = nn.Dropout(self.dropout_p)
        self.dropout6 = nn.Dropout(self.dropout_p)
        # map the output of LSTM to the output space
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        self.batch_size = input.shape[0]
        
        self.hidden1 = self.init_hidden()
        self.hidden2 = self.init_hidden()
        self.hidden3 = self.init_hidden()
        self.hidden4 = self.init_hidden()
        self.hidden5 = self.init_hidden()
        self.hidden6 = self.init_hidden()
        self.hidden7 = self.init_hidden()

        output, self.hidden1 = self.lstm_1(input.view(-1,1, self.input_size), self.hidden1)
        if self.verbose:
            output = self.dropout1(output)
        output_1 = output

        output, self.hidden2 = self.lstm_2(output, self.hidden2)
        if self.verbose:
            output = self.dropout2(output)
        output_2 = output

        output, self.hidden3 = self.lstm_3(output + output_1, self.hidden3)  # skip_connection 1
        if self.verbose:
            output = self.dropout3(output)
        output_3 = output

        output, self.hidden4 = self.lstm_4(output + output_2, self.hidden4)  # skip_connection 2
        if self.verbose:
            output = self.dropout4(output)
        output_4 = output

        output, self.hidden5 = self.lstm_5(output + output_3, self.hidden5)  # skip_connection 3
        if self.verbose:
            output = self.dropout5(output)
        output_5 = output

        output, self.hidden6 = self.lstm_6(output + output_4, self.hidden6)  # skip_connection 4
        if self.verbose:
            output = self.dropout6(output)
        output, self.hidden7 = self.lstm_7(output + output_5, self.hidden7)  # skip_connection 5
        
        output = self.out(output).view(self.batch_size, -1,1)
        # output = self.softmax(output)
        return output

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size // 2, device=device),
                torch.randn(2, 1, self.hidden_size // 2, device=device))
print(torch.cuda.is_available())


# In[8]:


BATCH_SIZE = 30


def validate(decoder, val_x, val_y, val_threshold=0.5):
    count = 0
    total = 0
    total_1 = 0

    # val_set = data_utils.TensorDataset(val_x, val_y)
    # val_loader=data_utils.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

    for i in range(len(val_x)):
        X = torch.from_numpy(val_x[i]).to(device).float()
        result = decoder(X).squeeze().cpu().detach().numpy()
        pr = [int(result[0] > 0.5)]
        for j in range(1, len(result) - 1):
            if result[j] > 0.5 and result[j] > result[j - 1] and result[j] > result[j + 1]:  # 回归
                pr.append(1)
            else:
                pr.append(0)
        pr.append(0)
        pr = np.array(pr).astype(int)
        Y = val_y[i].squeeze()
        gt = (Y == 1).astype(int)
        count += np.sum(gt * pr)  # TP
        total += np.sum(gt)  # TP+FP
        total_1 += np.sum(pr)  # TP+FN
    precision = count / (total + 0.0001)
    recall = count / (total_1 + 0.0001)  # Recall
    fscore = 2 * precision * recall / (precision + recall + 0.0001)
    return precision, recall, fscore

from copy import deepcopy

def penalty_loss(penalty, criterion, output, target):
    loss = 0
    batch_size = target.shape[0]
    for j in range(target.shape[0]):
        for i in range(target.shape[0]):
            if int(target[j, i]) == 1:
                loss += penalty[0] * criterion(output[j, i], target[j, i])
            else:
                loss += penalty[1] * criterion(output[j, i], target[j, i])
    return loss/batch_size

def train(input_tensor, target_tensor, decoder, decoder_optimizer, criterion, penalty = (1, 0.5)):
    decoder_optimizer.zero_grad()
    
    loss = 0
    
    
    decoder_output= decoder(input_tensor)
    
    #if verbose:
    #    print("prediction score:", decoder_output.squeeze().detach().cpu().numpy())
    #    print("ground truch:", target_tensor.squeeze().cpu().numpy())

    loss += penalty_loss(penalty, criterion, decoder_output.squeeze(0), target_tensor.float())

    loss.backward()
    
    decoder_optimizer.step()
    

    return loss.item()

def pad(vector, pad, dim=0):
    pad_size=list(vector.shape)
    #print(pad_size)
    pad_size[dim]=pad-vector.size(dim)
    #print(pad_size[dim])
    if pad_size[dim]<0:
        print("FATAL ERROR: pad_size=120 not enough!")
    return torch.cat([vector, torch.zeros(*pad_size).type(vector.type())], dim=dim)

def factorize(data_X, data_Y, size, batch_size, batch_length, augment= True):
    new_X = []
    new_Y = []
    for i in range(len(data_X)):
        X, Y = data_X[i], data_Y[i]
        flag = []
        for loc, j in enumerate(Y):
            if j == 1:
                flag.append(loc)
        prev = 0
        for j in range(4, len(flag), 5):
            new_X.append(pad(torch.from_numpy(X[prev:flag[j]]), batch_length))
            new_Y.append(pad(torch.from_numpy(Y[prev:flag[j]]), batch_length))
            prev = flag[j]
    train_X_new = new_X
    train_Y_new = new_Y
    pad_size = batch_length
    if augment:
        train_X_augment = []
        train_Y_augment = []
        for i, target in enumerate(train_Y_new):
            train_X_augment.append(pad(train_X_new[i], pad_size))
            train_Y_augment.append(pad(train_Y_new[i], pad_size))
            if augment_data:
                for direction in [-1, 1]:
                    for shift in range(1, 12):
                        for length in [0, 0.5]:
                            for silence_length in [0, 0.3]:
                                train_X_temp = (train_X_new[i]).clone()
                                train_X_temp[:, 2] += direction * shift
                                train_X_temp[:, 1] += silence_length
                                train_X_temp[:, 0] += length
                                train_X_augment.append(pad(train_X_temp, pad_size))
                                train_Y_augment.append(pad(train_Y_new[i], pad_size))
        train_X_new = torch.stack(train_X_augment)
        train_Y_new = torch.stack(train_Y_augment)
    return train_X_new, train_Y_new

    """for i in range(0, len(new_X), batch_size):
        if (i + batch_size) > (len(new_X) - 1):
            break
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            batch_x.append(pad(new_X[i + j], batch_length))
            batch_y.append(pad(new_Y[i + j], batch_length))
        batch_X.append(torch.stack(batch_x))
        batch_Y.append(torch.stack(batch_y))
    return batch_X, batch_Y"""
    print(torch.stack(new_X).shape, torch.stack(new_Y).shape)
    return torch.stack(new_X), torch.stack(new_Y)

def training_set_factorize(train_X, train_Y, pad_size, augment_data=True): #
    train_X_new = []
    train_Y_new = []
    for i, target in enumerate(train_Y):
        # print(i)
        total_len = 0
        one_list = []
        loc = 0
        for label in target:
            if (int(label[0]) == 1):
                one_list.append(loc)
            loc = loc + 1
        prev = 0
        # print(one_list)
        j = 0
        while (j < len(one_list)):
            if j == (len(one_list) - 1) and prev < one_list[j]: # 最后
                x_new = torch.from_numpy(train_X[i][prev:])
                y_new = torch.from_numpy(train_Y[i][prev:].reshape(-1))
                # print(x_new.size())
                # print(prev, one_list[j], j, "end")
                total_len += y_new.size(0)
                train_X_new.append(x_new)
                train_Y_new.append(y_new)
                j += 1
            elif (one_list[j] - prev) < 80:
                j += 1
            else:
                x_new = torch.from_numpy(train_X[i][prev:one_list[j - 1]])
                y_new = torch.from_numpy(train_Y[i][prev:one_list[j - 1]].reshape(-1))
                # print(prev, one_list[j-1], j)
                total_len += y_new.size(0)
                train_X_new.append(x_new)
                train_Y_new.append(y_new)
                prev = one_list[j - 1]
    train_X_augment = []
    train_Y_augment = []
    for i, target in enumerate(train_Y_new):
        train_X_augment.append(pad(train_X_new[i], pad_size))
        train_Y_augment.append(pad(train_Y_new[i], pad_size))
        if augment_data:
            for direction in [-1, 1]:
                for shift in range(1, 12):
                    for length in [0, 0.5]:
                        for silence_length in [0, 0.3]:
                            train_X_temp = (train_X_new[i]).clone()
                            train_X_temp[:, 2] += direction * shift
                            train_X_temp[:, 1] += silence_length
                            train_X_temp[:, 0] += length
                            train_X_augment.append(pad(train_X_temp, pad_size))
                            train_Y_augment.append(pad(train_Y_new[i], pad_size))
    train_X_new = torch.stack(train_X_augment)
    train_Y_new = torch.stack(train_Y_augment)
    return train_X_new, train_Y_new

def validation_set_factorize(train_X, train_Y, pad_size):
    train_X_new = []
    train_Y_new = []
    for i, target in enumerate(train_Y):
        train_X_new.append(pad(torch.from_numpy(train_X[i]), pad_size))
        train_Y_new.append(pad(torch.from_numpy(train_Y[i].reshape(-1)), pad_size))
    train_X_augment = []
    train_Y_augment = []
    # print(train_Y_new)
    for i, target in enumerate(train_Y_new):
        train_X_augment.append(train_X_new[i])
        train_Y_augment.append(train_Y_new[i])
    train_X_new = torch.stack(train_X_augment)
    train_Y_new = torch.stack(train_Y_augment)
    return train_X_new, train_Y_new
# In[9]:


import torch.utils.data as data_utils
class CrossValidator:
    def __init__(self, model, partition=1, decoder=None, batch_size=BATCH_SIZE, batch_length = 120, epochs=10, lr=1e-2,
                 augment_data=0, print_every = 1000, plot_every = 100, gamma = 0.1):
        self.model=model
        self.data_X = []
        self.data_Y = []
        self.augment_data_size=augment_data
        with open("/gpfsnyu/home/yz6492/melody_seg/data/essen_binary.pkl", "rb") as f:
            data= pickle.load(f)
            for i in range(len(data)):
                self.data_X = data["X"]
                self.data_Y = data["Y"]
            
        
        self.data_size = len(self.data_X)
        self.partition=partition
        self.decoder=decoder
        self.train_X=[]
        self.train_Y=[]
        self.val_X=[]
        self.val_Y=[]
        self.precision_history=[]
        self.recall_history=[]
        self.loss_history=[]
        self.best_acc = 0
        self.batch_size=batch_size
        self.batch_length=batch_length
        self.epochs=epochs
        self.lr=lr
        self.gamma = gamma
        self.print_every = print_every
        self.plot_every = plot_every
        
        
    def create_data(self):
        prec = [int(i * self.data_size) for i in [0, 0.8, 0.9, 1]]
        train_X = [np.array(line) for line in self.data_X[prec[0]: prec[1]]]
        val_X = [np.array(line) for line in self.data_X[prec[1]: prec[2]]]
        test_X = [np.array(line) for line in self.data_X[prec[2]: prec[3]]]
        train_Y = [np.array(line) for line in self.data_Y[prec[0]: prec[1]]]
        val_Y = [np.array(line) for line in self.data_Y[prec[1]: prec[2]]]
        test_Y = [np.array(line) for line in self.data_Y[prec[2]: prec[3]]]
        return train_X, train_Y, val_X, val_Y, test_X, test_Y
    
    def tensorize(self, p):
        p=np.array(p)
        p=torch.from_numpy(p).float()
        p=p.to(device)
        return p
    
    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def timeSince(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def compute(self):
        start = time.time()
        temptrain_X, temptrain_Y, tempval_X, tempval_Y , temptest_X, temptest_Y = self.create_data()
        print(len(temptrain_X), len(temptrain_Y), len(tempval_X), len(tempval_Y))
        sys.stdout.flush()
        self.train_X, self.train_Y = training_set_factorize(temptrain_X, temptrain_Y, self.batch_length)
        self.val_X, self.val_Y = training_set_factorize(tempval_X, tempval_Y, self.batch_length)
        self.test_X, self.test_Y = validation_set_factorize(temptest_X, temptest_Y, valid_length)
        self.val_X, self.val_Y = tempval_X, tempval_Y
        train_set = data_utils.TensorDataset(self.train_X, self.train_Y)
        train_loader=data_utils.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
        # print(len(train_loader))

        cur_model = deepcopy(self.model).to(device)

        optimizer = optim.SGD(cur_model.parameters(), lr = self.lr, weight_decay = 1e-5)

        criterion = nn.SmoothL1Loss()

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = self.gamma)

        print_loss_total = 0
        plot_loss_total = 0

        for j in range(self.epochs):
            for num, (train_X, train_Y)in enumerate(train_loader):
                input_tensor = train_X.to(device).float()
                target_tensor = train_Y.to(device).float()
                loss = train(input_tensor, target_tensor, cur_model, decoder_optimizer=optimizer, criterion= criterion)
                # print(loss)
                print_loss_total += loss
                plot_loss_total += loss

                if num%self.plot_every == 0:
                    plot_loss_avg = plot_loss_total / self.plot_every
                    plot_loss_total = 0
                    self.loss_history.append(plot_loss_avg)

                if (num) % self.print_every == 0:
                    acc, one_acc, score = validate(cur_model, self.val_X, self.val_Y)
                    self.precision_history.append(acc[:-1])
                    self.recall_history.append(one_acc[:-1])
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    print("epoch %i"%j)
                    p = self.timeSince(start, (1 + num+j*len(train_loader)) / (self.epochs * len(train_loader)))
                    print('%s (%d %d%%) %.4f' % (p, num + 1, (num + 1) / (self.epochs * len(train_loader)) * self.print_every,
                                                 print_loss_avg))
                    print("validation accuracy:", acc)
                    print("validation prediction accuracy:", one_acc)
                    if(score > self.best_acc):
                        torch.save(cur_model.state_dict(), 'Bi-LSTM_best.pt')
                        self.best_acc = score
                    print("best_score:", self.best_acc)
                    sys.stdout.flush()

            scheduler.step()
            name = 'lstm_binary_train_epoch' + str(j) + '.pt'
            torch.save(cur_model.state_dict(), name)
            print("completed: ", float(j) / float(self.epochs))
                #torch.save(cur_model.state_dict(), '/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN(cv){}-{}.pt'.format(i,j))
        acc, one_acc, score = validate(cur_model, self.test_X, self.test_Y)
                
        return self.loss_history, self.acc, one_acc



# In[10]:


input_size = 3
augmented_size = 32
hidden_size = 256
output_size = 1
batch_size = 30
batch_length = 120
valid_length = 400
model = DecoderRNN(input_size, augmented_size, hidden_size, output_size, dropout_p = 0).to(device)
total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print({'Total': total_num, 'Trainable': trainable_num})
sys.stdout.flush()
cv = CrossValidator(model, partition=1, epochs=5, batch_size = batch_size, batch_length=batch_length, augment_data=0, print_every = 20)
# losses, precision, recall = cv.compute()
#
# d = {"p": precision, "r": recall, "f": 2*precision*recall/(precision+recall)}
# print(d)
# sys.stdout.flush()
# dic = {}
# dic["loss"] = losses
# dic["precision"] = precision
# dic["recall"] = recall

is_validate= True
if is_validate:
    _, _, _, _, temptest_X, temptest_Y = cv.create_data()
    test_X, test_Y = validation_set_factorize(temptest_X, temptest_Y, valid_length)
    test_X, test_Y = test_X.numpy(), test_Y.numpy()
    model.load_state_dict(torch.load("/gpfsnyu/home/yz6492/melody_seg/lstm_binary_train_epoch4.pt"))
    precision, recall, fscore = validate(model, test_X, test_Y)
    d = {"p": precision, "r": recall, "f": fscore}
    print(d)

#f = open("/home/yiqin/2018summer_project/saved_model/Bi-LSTM-CNN_losses(cv-l2).pkl", "wb")
#pickle.dump(dic, f)


# In[20]:


def evaluate(model, val_x):
    prediction = []
    for i in range(len(val_x)):
        X = val_x[i].to(device).float()
        result = model(X).squeeze().cpu().detach().numpy()
        pr = []
        pr.append(1)
        for j in range(1, len(result) - 1):
            if result[j] > 0.5 and result[j] > result[j-1] and result[j] > result[j+1]:
                pr.append(1)
            else:
                pr.append(0)
        pr.append(0)
        prediction.append(np.array(pr))
    return prediction



# In[ ]:


# input_size = 3
# augmented_size = 32
# hidden_size = 256
# output_size = 1
# model = DecoderRNN(input_size, augmented_size, hidden_size, output_size).to(device)
# # model.load_state_dict(torch.load("/gpfsnyu/scratch/yz6492/melody_seg/Bi-LSTM-CNN1.pt"))
# prediction = evaluate(model, val_X)
# acc, one_acc = validate(model, val_X,val_Y, 0.52)
# print(acc, one_acc)


# In[ ]:

#
# import pickle
# f = open("/gpfsnyu/scratch/yz6492/melody_seg/BiLSTM-CNN.pkl", "wb")
# pickle.dump(prediction, f)

