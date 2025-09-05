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
from MyformerModel.model import Myformer
from datasets.softfall_dataset import MySoftfallData
from datasets.sisfall_dataset import MySisfallData
from datasets.kfall_dataset import MyKfallData
from datasets.cgubes_dataset import MyCGUBESData
from datasets.softfall_dataset_basic import MySoftfallDataBasic
from datasets.softfall_dataset_complex import MySoftfallDataComplex

from paras import get_parameters

seed(10)

scaler_minus1_1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaler_0_1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler_1_2 = preprocessing.MinMaxScaler(feature_range=(1, 2))
args = get_parameters()


class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()

    def forward(self, x):
        return x.mean(dim=-2)


if args.dataset_name == "softfall":
    GCT_row = torch.tensor(args.softfall_38)
elif args.dataset_name == "kfall":
    GCT_row = torch.tensor(args.kfall_38)
elif args.dataset_name == "sisfall":
    GCT_row = torch.tensor(args.sisfall_38)
elif args.dataset_name == "cgubes":
    GCT_row = torch.tensor(args.cgubes_38)
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
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        dl = self.data_linear(x)
        output = torch.mul(self.weights_row, dl)
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
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, d_model, num_classes):
        super(CNN, self).__init__()
        self.bn = nn.BatchNorm1d(args.axis_num)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dp = nn.Dropout(p=0.8)
        self.gap = GAP()
        self.fc = nn.Linear(12, num_classes)

    def forward(self, x):
        out = self.bn(x)
        out = self.conv1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.dp(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# LSTM
class LSTM(nn.Module):
    def __init__(self, input_size):
        super(LSTM, self).__init__()
        self.bn = nn.BatchNorm1d(args.axis_num)
        self.dp1 = nn.Dropout(p=0.8)
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True)
        self.dp2 = nn.Dropout(p=0.8)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.gap = GAP()
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.bn(x)
        x = self.dp1(x)
        x = self.lstm1(x)
        x = x[0]
        x = self.dp2(x)
        x = self.lstm2(x)
        out = x[0]
        out = self.gap(out)
        out = self.fc(out)
        return out


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # out: tensor of shape (batch_size, num_classes)
        return out


class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=2):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn_block1 = CNN_Block(input_size, 32)
        self.cnn_block2 = CNN_Block(32, 64)
        self.cnn_block3 = CNN_Block(64, 32)

        self.lstm = nn.LSTM(9, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        # x = self.cnn_block4(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Final time step output from LSTM
        out = out[:, -1, :]
        # Fully connected layer
        # out = self.gap(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dp(out)
        out = self.fc(out)
        return out


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


class Conv1D_Block(nn.Module):
    def __init__(self, c_in, distill=args.isDistill):
        super(Conv1D_Block, self).__init__()
        self.distill = distill
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.Conv1d = nn.Conv1d(in_channels=c_in,
                                out_channels=c_in,
                                kernel_size=3,
                                padding=padding,
                                padding_mode='replicate')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.Conv1d(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class Conv2D_Block(nn.Module):
    def __init__(self, c_in, distill=args.isDistill):
        super(Conv2D_Block, self).__init__()
        self.distill = distill
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.Conv2d = nn.Conv2d(in_channels=c_in,
                                out_channels=c_in,
                                kernel_size=3,
                                padding=padding,
                                padding_mode='replicate')
        self.norm = nn.BatchNorm2d(1)
        self.activation = nn.ReLU(inplace=True)
        if self.distill:
            self.Pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        batch_size_raw, input_height_raw, input_width_raw = x.shape
        channel_raw = 1
        x = x.view(batch_size_raw, channel_raw, input_height_raw, input_width_raw)
        x = self.Conv2d(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x.view(batch_size_raw, input_height_raw, input_width_raw)
        if self.distill:
            x = x.permute(0, 2, 1)
            x = self.Pool(x)
            x = x.transpose(1, 2)
        return x


class CausalFall(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        super(CausalFall, self).__init__()
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=16)
        self.transformer_encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=2)

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=16)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=2)

        self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=16)
        self.transformer_encoder3 = nn.TransformerEncoder(self.encoder_layer3, num_layers=2)

        if args.isConv == 2:
            self.conv1 = Conv2D_Block(1)
            self.conv2 = Conv2D_Block(1)
        elif args.isConv == 1:
            self.conv1 = Conv1D_Block(d_model)
            self.conv2 = Conv1D_Block(d_model)

        self.fc1 = nn.Linear(d_model, 8)
        self.ln = nn.LayerNorm(8)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(p=0.1)
        self.gap = GAP()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        if args.isConv:
            x = self.transformer_encoder1(x)
            x = self.conv1(x)
            x = self.transformer_encoder2(x)
            x = self.conv2(x)
            x = self.transformer_encoder3(x)
        else:
            x = self.transformer_encoder1(x)
            x = self.transformer_encoder2(x)
            x = self.transformer_encoder3(x)

        x = self.fc1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.gap(x)
        x = self.fc2(x)
        return x


now = datetime.datetime.now()
timeStr = now.strftime("%Y-%m-%d_%H-%M-%S")

if args.model_name == "Myformer" or args.model_name == "CausalFall":
    save_dir = f"{args.save_path}/{args.dataset_name}_{args.model_name}_isCE{args.isCE}_isPE{args.isPE}_isConv{args.isConv}_isDistill{args.isDistill}_isCF1_isHetero{args.isHetero}_trial{args.trial}_{timeStr}"
else:
    save_dir = f"{args.save_path}/{args.dataset_name}_{args.model_name}_isCE{args.isCE}_isPE{args.isPE}_isCF1_isHetero{args.isHetero}_trial{args.trial}_{timeStr}"
if not os.path.exists(save_dir):
    print(save_dir)
    os.makedirs(save_dir)

model = nn.Sequential()

if args.isCE == 1:
    if args.model_name != "Myformer":
        model.add_module("Causal Embedding", weighted_pooling)


if args.isPE == 1:
    need_PE = False
    model.add_module("Positional Embedding", PositionalEncoding(args.d_model))
else:
    need_PE = False

if args.model_name == "CNN":
    model.add_module("CNN", CNN(input_size=args.axis_num, hidden_size=64, d_model=args.d_model, num_classes=2))
elif args.model_name == "LSTM":
    model.add_module("LSTM", LSTM(input_size=args.d_model))
elif args.model_name == "BiLSTM":
    model.add_module("BiLSTM", BiLSTM(input_size=args.d_model, hidden_size=32, num_layers=2, num_classes=2))
elif args.model_name == "ConvLSTM":
    model.add_module("ConvLSTM", ConvLSTM(input_size=args.axis_num, hidden_size=32, num_layers=2, num_classes=2))
elif args.model_name == "Transformer":
    model.add_module("Transformer", Transformer(d_model=args.d_model, nhead=args.n_head))
elif args.model_name == "Myformer":
    model.add_module("Myformer", Myformer(input_size=args.d_model, hidden_size=args.projected_dim, output_size=2, n_heads=args.n_head, axis_num=args.axis_num, e_layers=3))
elif args.model_name == "CausalFall":
    model.add_module("CausalFall", CausalFall(d_model=args.d_model, nhead=args.n_head))

print(model)

if args.isHetero == 0:
    if args.dataset_name == "softfall":
        my_dataset = MySoftfallData(need_pe=need_PE)
    elif args.dataset_name == "kfall":
        my_dataset = MyKfallData(need_pe=need_PE)
    elif args.dataset_name == "sisfall":
        my_dataset = MySisfallData(need_pe=need_PE)
    elif args.dataset_name == "cgubes":
        my_dataset = MyCGUBESData(need_pe=need_PE)

    if args.dataset_name == "softfall":
        train_size = int(0.6 * len(my_dataset))
    elif args.dataset_name == "kfall":
        train_size = int(0.6 * len(my_dataset))
    elif args.dataset_name == "sisfall":
        train_size = int(0.8 * len(my_dataset))
    elif args.dataset_name == "cgubes":
        train_size = int(0.8 * len(my_dataset))

    test_size = len(my_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=my_dataset, lengths=[train_size, test_size])

elif args.isHetero == 1:
    train_dataset = MySoftfallDataBasic(need_pe=need_PE)
    test_dataset = MySoftfallDataComplex(need_pe=need_PE)
    # train_size = int(len(my_dataset_basic))
    # test_size = int(len(my_dataset_complex))
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset=my_dataset, lengths=[train_size, test_size])
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset=my_dataset, lengths=[train_size, test_size])

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
            """训练数据：用哪些轴，是否滤波"""
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
        for step, (train_signal, _, train_label) in enumerate(train_loader):
            cf_signal = torch.zeros_like(train_signal)
            cf_signal = cf_signal.to(device)

            train_label = train_label.to(device)

            train_logits = net(cf_signal)

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

    if epoch == 0:
        trail_train_loss = trail_train_loss[1:]
        trail_test_accuracy = trail_test_accuracy[1:]
        testLabels = testLabels[1:]
        predLabels = predLabels[1:]

    if epoch > 0 and (epoch + 1) % 200 == 0:
        torch.save(model.state_dict(), f"{save_dir}/modelWeights_epoch{epoch+1}.pth")

# total = sum([param.nelement() for param in net.parameters()])
# print("Number of parameter: %.2fM" % (total / 1e6))

index = 0
for (test_signal, _, test_label) in test_loader:
    if index == 0:
        test_cat = test_signal
        index += 1
    else:
        test_cat = torch.cat([test_cat, test_signal], dim=0)

if torch.cuda.is_available():
    test_cat = test_cat.cuda()

flops, params = profile(net, inputs=(test_cat,))
flops, params = clever_format([flops, params], "%.3f")
print("---------------------------------------------------------")
print("Number of flops: ", flops)
print("Number of params: ", params)
print("---------------------------------------------------------")

torch.save(trail_train_loss, f"{save_dir}/loss.pth")
torch.save(trail_test_accuracy, f"{save_dir}/accuracy.pth")
torch.save(testLabels, f"{save_dir}/testLabels.pth")
torch.save(predLabels, f"{save_dir}/predLabels.pth")

df = pd.DataFrame()
df.to_csv(f"{save_dir}/record_Acc{Acc}_Time{Time}_Paras{params}_Flops{flops}.csv")
