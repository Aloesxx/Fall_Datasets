from paras import get_parameters
import torch
import torch.nn as nn
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from torchvision.transforms import Normalize
import torch.nn.functional as F
import pandas as pd
import seaborn as sns

args = get_parameters()


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class GetPthData():
    def __init__(self, pth_data_dir, file_name):
        self.dir = pth_data_dir
        self.file_name = file_name

    def get_raw(self):
        full_dir = os.path.join(self.dir, self.file_name)
        raw_data = torch.load(full_dir)
        trans_data = torch.transpose(raw_data, 1, 2)
        return trans_data

    # def down_sample_to_40Hz_and_select_first_three_dimension(data):
    #     return data[:, ::5, 0:3]

    def down_sample(self, step=args.step_down):
        data = self.get_raw()
        return data[:, :, ::step]

    def three_second_window_data(self):
        return None


class ButterWorth():
    def __init__(self, data):
        self.root = r".\static\qs_10383_1.csv"
        self.data = data
        self.axisX = np.arange(1, self.data.shape[1] + 1)
        self.xLabel = 'Samples'
        self.yLabel = 'Accelerator (g)'
        self.lineLegend = ['Accelerator_X', 'Accelerator_Y', 'Accelerator_Z', 'Filtered_X', 'Filtered_Y', 'Filtered_Z']
        self.lineColor = ['blue', 'teal', 'blueviolet', 'greenyellow', 'cyan', 'darkorange']

    def lowPass(self):
        cut = 5.0
        fs = 25.0
        wn = 2 * cut / fs
        b, a = signal.butter(5, wn, 'lowpass')
        filtered = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            filtered[i] = np.array(signal.filtfilt(b, a, self.data[i]))
        return filtered

    def highPass(self):
        cut = 8.0
        fs = 25.0
        wn = 2 * cut / fs
        b, a = signal.butter(5, wn, 'highpass')
        filtered = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            filtered[i] = np.array(signal.filtfilt(b, a, self.data[i]))
        return filtered

    def bandPass(self):
        lowCut = 5.0
        highCut = 8.0
        fs = 25.0
        wn_low = 2 * lowCut / fs
        wn_high = 2 * highCut / fs
        b, a = signal.butter(5, [wn_low, wn_high], 'bandpass')
        filtered = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            filtered[i] = np.array(signal.filtfilt(b, a, self.data[i]))
        return filtered

    def pltFigure(self, data, figSaveRoot, figName):
        fig = plt.figure(figsize=(9, 4), dpi=200)
        ax = plt.axes()
        for i in range(data.shape[0]):
            axis_data = np.squeeze(data[i])
            if i < 3:
                ax.plot(self.axisX, axis_data, color=self.lineColor[i], label=self.lineLegend[i])
            else:
                ax.plot(self.axisX, axis_data, color=self.lineColor[i], label=self.lineLegend[i])

        ax.set(xlabel=self.xLabel, ylabel=self.yLabel)
        ax.legend()
        figName = figName + '.png'
        figRoot = os.path.join(figSaveRoot, figName)
        plt.savefig(figRoot)
        plt.close()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # pos/10000^(2i/dmodel)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def modalEmbedding(modal_num, batch_data):
    modal_types = torch.zeros(3)
    for i in range(modal_num - 1):
        modal_types = torch.cat([modal_types, torch.ones(3) + i])
    modal_embedded = torch.cat([modal_types.view(1, modal_num * 3, 1).expand(batch_data.shape[0], -1, -1), batch_data],
                               dim=2)
    return modal_embedded


def modalEmbeddingAblation(modal_num, batch_data):
    modal_types = torch.zeros(3) - 1
    for i in range(modal_num - 1):
        modal_types = torch.cat([modal_types, torch.zeros(3) - 1])
    modal_embedded = torch.cat([modal_types.view(1, modal_num * 3, 1).expand(batch_data.shape[0], -1, -1), batch_data],
                               dim=2)
    return modal_embedded
