import torch
from utils import GetPthData
from paras import get_parameters
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from sklearn import preprocessing
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

scaler_minus1_1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaler_0_1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler_1_2 = preprocessing.MinMaxScaler(feature_range=(1, 2))
args = get_parameters()


GCT_row = torch.tensor([2606., 3428., 2653., 4419., 4963., 4547., 2596., 3455., 2801.])
GCT_row = scaler_1_2.fit_transform(GCT_row.reshape(-1, 1))
GCT_row = torch.as_tensor(GCT_row).float()

GC_kernel_index = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8]


def SMV(batch_data):
    if batch_data.ndim == 2:
        axis_dim = 0
    elif batch_data.ndim == 3:
        axis_dim = 1
    else:
        return "Input shape error."
    smv_pow2 = torch.pow(batch_data, 2)  # [x, 3, 1600]
    smv_sum = torch.sum(smv_pow2, dim=axis_dim)  # [x, 1600]
    smv = torch.sqrt(smv_sum)  # [x, 1600]
    arglag = 50
    idx_falling_end = torch.argmax(smv[:, arglag:], dim=axis_dim)  # [x]
    idx_falling_end = idx_falling_end + arglag
    idx_falling_start = []
    for i in range(idx_falling_end.shape[0]):
        fallingData = np.squeeze(smv[i])
        varlen = idx_falling_end[i]
        varlist = []
        idxTemp = 0
        lag = 40
        for j in (range(lag, varlen)):
            var_list_small = fallingData[j:j + 5].var()
            var_list_media = fallingData[j:j + 10].var()
            varlist.append((var_list_small + var_list_media) / 2)
            m = j - lag
            if varlist[m] > 0.003:
                idxTemp = m + lag
                for k in reversed(range(m)):
                    if varlist[k] < 0.0011:
                        idxTemp = k + lag
                        break
                idx_falling_start.append(idxTemp)
                break
            elif j == varlen-1:
                # 找不到很波动的点
                idx_falling_start.append(varlen-25)
                break

    idx_falling_start = torch.from_numpy(np.array(idx_falling_start))
    return idx_falling_start, idx_falling_end, smv, smv_sum


class GetFeaWinFromDataset():
    def __init__(self, multi_axis_dataset, pre_len, front_len, rear_len):
        acc = multi_axis_dataset[:, 0:3, :]
        smv_start_idx, smv_max_idx, smv_sqrt, smv_sum = SMV(acc)
        self.batch_data = multi_axis_dataset
        self.target_idx = smv_max_idx - pre_len
        self.falling_start_idx = smv_start_idx
        self.falling_len = args.falling_len
        self.front_len = front_len
        self.rear_len = rear_len

    def getGCTWinByTgtIdx(self):
        if self.batch_data.ndim == 2:
            if (self.falling_len >= 0) and (self.falling_len <= self.target_idx) and (self.target_idx + self.falling_len + 1 <= self.batch_data.shape[-1]):
                self.batch_data[..., 0:(self.falling_len * 2 + 1)] = self.batch_data[..., (self.target_idx - self.falling_len):(self.target_idx + self.falling_len + 1)].clone()
            else:
                self.batch_data = torch.zeros_like(self.batch_data)
            return self.batch_data[..., 0:(self.falling_len * 2 + 1)]

        if self.batch_data.ndim == 3:
            for i in range(self.batch_data.shape[0]):
                if (self.falling_len >= 0) and (self.falling_len <= self.target_idx[i]) and (self.target_idx[i] + self.falling_len + 1 <= self.batch_data[i].shape[-1]):
                    self.batch_data[i, :, 0:(self.falling_len * 2 + 1)] = self.batch_data[i, :, (self.target_idx[i] - self.falling_len):(self.target_idx[i] + self.falling_len + 1)].clone()
                else:
                    self.batch_data[i, ...] = torch.zeros_like(self.batch_data[i, ...])

            non_zero_rows = torch.abs(self.batch_data[:, 0, :]).sum(dim=-1) > 0
            self.batch_data = self.batch_data[non_zero_rows]
            return self.batch_data[..., 0:(self.falling_len * 2 + 1)]

    def getFeaWinByTgtIdx(self):
        if self.batch_data.ndim == 2:

            for i in range(self.batch_data.shape[0]):
                self.batch_data[i, :] = torch.as_tensor(scaler_minus1_1.fit_transform(self.batch_data[i, :].reshape(-1, 1)).reshape(1, -1))
            if (self.front_len >= 0) and (self.rear_len >= 0) and (self.front_len <= self.target_idx) and (self.target_idx + self.rear_len + 1 <= self.batch_data.shape[-1]):
                self.batch_data[..., 0:(self.front_len + self.rear_len + 1)] = self.batch_data[..., (self.target_idx - self.front_len):(self.target_idx + self.rear_len + 1)].clone()
            else:
                self.batch_data = torch.zeros_like(self.batch_data)
            return self.batch_data[..., 0:(self.front_len + self.rear_len + 1)]

        if self.batch_data.ndim == 3:
            for i in range(self.batch_data.shape[0]):
                for j in range(self.batch_data.shape[1]):
                    self.batch_data[i, j, :] = torch.as_tensor(scaler_minus1_1.fit_transform(self.batch_data[i, j, :].reshape(-1, 1)).reshape(1, -1))

                if (self.front_len >= 0) and (self.rear_len >= 0) and (self.front_len <= self.target_idx[i]) and (self.target_idx[i] + self.rear_len + 1 <= self.batch_data[i].shape[-1]):
                    self.batch_data[i, :, 0:(self.front_len + self.rear_len + 1)] = self.batch_data[i, :, (self.target_idx[i] - self.front_len):(self.target_idx[i] + self.rear_len + 1)].clone()
                else:
                    self.batch_data[i, ...] = torch.zeros_like(self.batch_data[i, ...])

            non_zero_rows = torch.abs(self.batch_data[:, 0, :]).sum(dim=-1) > 0
            self.batch_data = self.batch_data[non_zero_rows]
            output = self.batch_data[..., 0:(self.front_len + self.rear_len + 1)]
            return output

    def setWgtByTgtIdx(self, input):
        batch_size_raw, input_height_raw, input_width_raw = input.shape
        channel_raw = 1
        input = input.view(batch_size_raw, channel_raw, input_height_raw, input_width_raw)
        # input = replicate_padding(input, padding=1)
        batch_size, channel, input_height, input_width = input.shape
        kernel_size = 3
        kernel = torch.zeros(kernel_size, kernel_size)
        output_height = input_height - kernel_size + 1
        output_width = input_width - kernel_size + 1
        output = torch.zeros(batch_size, channel, output_height, output_width)
        for h in range(output_height):
            for i_k in range(kernel_size):
                for j_k in range(kernel_size):
                    kernel[i_k, j_k] = GCT_row[GC_kernel_index[h + i_k]].cuda()
            for w in range(output_width):
                output[:, :, h, w] = torch.einsum('abcd,cd->ab',
                                                  input[:, :, h:h + kernel_size, w:w + kernel_size].float(),
                                                  kernel.float())
        output = output.view(batch_size_raw, input_height_raw, input_width_raw)
        return output

    def getGCTWindow(self):
        fea_window = self.getGCTWinByTgtIdx()
        return fea_window

    def getWindow(self):
        fea_window = self.getFeaWinByTgtIdx()
        return fea_window
