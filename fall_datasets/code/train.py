import os
import math
import time
import torch
from thop import clever_format, profile
from torch import nn
import numpy as np
import datetime
from random import *
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing
from visdom import Visdom
from utils import AverageMeter, ProgressMeter
from datasets.slowfall_dataset import MySlowfallData

from paras import get_parameters

# seed(10)

scaler_minus1_1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaler_0_1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler_1_2 = preprocessing.MinMaxScaler(feature_range=(1, 2))
args = get_parameters()

class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()

    def forward(self, x):
        return x.mean(dim=-2)

if args.dataset_name == "slowfall":
    GCT_row = torch.tensor(args.slowfall_38)
GCT_row = scaler_1_2.fit_transform(GCT_row.reshape(-1, 1))
GCT_mat = np.zeros([args.axis_num, args.axis_num])
for i in range(args.axis_num):
    for j in range(args.axis_num):
        GCT_mat[j][i] = GCT_row[j]
GCT_mat = scaler_1_2.fit_transform(GCT_mat)
GCT_row = torch.as_tensor(GCT_row).float()
if torch.cuda.is_available():
    GCT_row = GCT_row.cuda()
GCT_mat = torch.as_tensor(GCT_mat).float()


class WeightedPoolingLayer(nn.Module):
    def __init__(self, input_size, axis_dim):
        super(WeightedPoolingLayer, self).__init__()
        self.GCT_linear = nn.Linear(axis_dim, axis_dim)
        self.data_linear = nn.Linear(input_size, input_size)
        self.weights_row = nn.Parameter(torch.Tensor(axis_dim), requires_grad=True)
        # self.weights_mat = nn.Parameter(torch.Tensor(axis_dim, axis_dim), requires_grad=True)
        # self.weights_conv = nn.Parameter(torch.Tensor(axis_dim, axis_dim), requires_grad=True)
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        dl = self.data_linear(x)
        # wl = self.GCT_linear(self.weights_row)
        output = torch.mul(self.weights_row, dl)
        # output = torch.mul(self.weights_row, x)
        linear_output = self.linear2(nn.ReLU(inplace=True)(self.linear1(output)))
        output = self.layer_norm(linear_output)
        return output


weighted_pooling = WeightedPoolingLayer(input_size=args.d_model, axis_dim=args.axis_num)
weighted_pooling.weights_row.data = GCT_row


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe[:, :x.size(1), :]
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=16)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.fc1 = nn.Linear(72, 8)
        self.ln = nn.LayerNorm(8)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(p=0.1)
        self.gap = GAP()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.gap(x)
        x = self.fc2(x)
        return x


now = datetime.datetime.now()
timeStr = now.strftime("%Y-%m-%d_%H-%M-%S")

""" 
python -m visdom.server
"""
model = nn.Sequential()

if args.isCE == 1:
    model.add_module("Causal Embedding", weighted_pooling)

if args.isPE == 1:
    need_PE = True
    model.add_module("Positional Embedding", PositionalEncoding(args.d_model))
else:
    need_PE = False

if args.model_name == "Transformer":
    model.add_module("Transformer", Transformer(d_model=args.d_model, nhead=args.n_head))

print(model)

if args.isHetero == 0:
    if args.dataset_name == "slowfall":
        my_dataset = MySlowfallData(need_pe=need_PE)

    if args.dataset_name == "slowfall":
        train_size = int(0.6 * len(my_dataset))

    test_size = len(my_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=my_dataset, lengths=[train_size, test_size])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=0, pin_memory=True, drop_last=False)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

net = model.to(device)

init_lr = 1e-3

def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

# optimizer = torch.optim.SGD(net.parameters(), init_lr, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(net.parameters(), init_lr, weight_decay=1e-4)

criteon = nn.CrossEntropyLoss().to(device)

viz = Visdom()
viz.line([0.], [0.], win='train', opts=dict(title='train loss'))
viz.line([0.], [0.], win='test', opts=dict(title='test acc'))

global_loss = 0

trail_train_loss = torch.tensor(-1)
trail_test_accuracy = torch.tensor(-1)

testLabels = torch.tensor(2).to(device)
predLabels = torch.tensor(2).to(device)

Acc = ""
Time = ""

# epochs = args.epochs
gap = int(200)
epochs = int(gap*2 + gap/2)
loss_temp_temp = []
for epoch in range(epochs):
    adjust_learning_rate(optimizer, init_lr, epoch, epochs)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    net.train()
    end = time.time()

    if epoch < gap or epoch >= gap:
        loss_temp = []
        for step, (train_signal, _, train_label) in enumerate(train_loader):
            # print("---------------------------------------------------------")
            # print("step: ", step)
            # print("---------------------------------------------------------")
            # train_signal = train_signal[:, 0:6, ...].to(device)
            train_signal = train_signal.to(device)
            train_label = train_label.to(device)

            train_logits = net(train_signal)
            loss_temp.append(train_logits)
            train_loss = criteon(train_logits, train_label)
            losses.update(train_loss.item(), train_signal.size(0))

            global_loss = train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if step % 10 == 0:
                progress.display(step)

        loss_temp_temp.append(loss_temp)
    elif gap <= epoch < gap * 2:
        # print(len(loss_temp_temp))
        # print(len(loss_temp_temp[0]))
        for step, (train_signal, _, train_label) in enumerate(train_loader):
            # train_signal = train_signal[:, 0:6, ...].to(device)
            cf_signal = torch.zeros_like(train_signal)
            cf_signal = cf_signal.to(device)

            train_label = train_label.to(device)
            train_logits = net(cf_signal)

            # CounterFactual
            label_origin = loss_temp_temp[epoch-gap][step]
            label_causal = label_origin - train_logits

            train_loss = criteon(label_causal, train_label)
            losses.update(train_loss.item(), train_signal.size(0))
            global_loss = train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if step % 10 == 0:
                progress.display(step)

    viz.line([global_loss], [epoch], win='train', update='append')
    trail_train_loss = torch.hstack([trail_train_loss, torch.tensor(global_loss)])

    net.eval()
    eval_start_time = time.time()
    with torch.no_grad():
        test_loss = 0
        correct = 0

        for (test_signal, _, test_label) in test_loader:
            # test_signal = test_signal[:, 0:6, ...].to(device) # a,g
            test_signal = test_signal.to(device)  # a,g,m
            test_label = test_label.to(device)

            testLabels = torch.hstack([testLabels, test_label])

            test_logits = net(test_signal)
            test_loss += criteon(test_logits, test_label)

            pred = test_logits.argmax(dim=1)

            predLabels = torch.hstack([predLabels, pred])

            correct += pred.eq(test_label).float().sum().item()

        eval_end_time = time.time() - eval_start_time
        Time = str(round(eval_end_time, 3))
        accuracy = correct / len(test_loader.dataset)
        Acc = str(round(accuracy, 5))
        print("---------------------------------------------------------")
        print('\n{} \nTest set: Accuracy: {}/{} ({:.3f}%)\n'.format(
            eval_end_time, correct, len(test_loader.dataset),
            100. * accuracy))
        print("---------------------------------------------------------")
        viz.line([accuracy], [epoch], win='test', update='append')

        trail_test_accuracy = torch.hstack([trail_test_accuracy, torch.tensor(accuracy)])
