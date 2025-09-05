from paras import get_parameters
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from features import GetFeaWinFromDataset
from utils import GetPthData, ButterWorth, PositionalEncoding, modalEmbeddingAblation, modalEmbedding


class AD2Raw():
    def __init__(self, ori_data, ori_name):
        self.ori_data = ori_data
        # sisfall
        if ori_name == 'sisfall':
            self.sisfall_acc1_range = torch.tensor(16)
            self.sisfall_acc1_resolution = torch.tensor(13)
            self.sisfall_gyr_range = torch.tensor(2000)
            self.sisfall_gyr_resolution = torch.tensor(16)
            self.sisfall_acc2_range = torch.tensor(8)
            self.sisfall_acc2_resolution = torch.tensor(14)
        else:
            self.default_acc_range = torch.tensor(16)
            self.default_acc_resolution = torch.tensor(13)
            self.default_gyr_range = torch.tensor(2000)
            self.default_gyr_resolution = torch.tensor(16)
            self.default_mag_range = torch.tensor(8)
            self.defaultl_mag_resolution = torch.tensor(14)

    def fromSisfall(self):
        self.ori_data[:, 0:3, :] = ((2 * self.sisfall_acc1_range)/torch.pow(2, self.sisfall_acc1_resolution)) * self.ori_data[:, 0:3, :]
        self.ori_data[:, 3:6, :] = ((2 * self.sisfall_gyr_range)/torch.pow(2, self.sisfall_gyr_resolution)) * self.ori_data[:, 3:6, :]
        self.ori_data[:, 6:9, :] = ((2 * self.sisfall_acc2_range)/torch.pow(2, self.sisfall_acc2_resolution)) * self.ori_data[:, 6:9, :]
        return self.ori_data


class MySisfallData(Dataset):
    def __init__(self, need_pe):
        super(MySisfallData, self).__init__()
        args = get_parameters()
        adl_12_raw = GetPthData(pth_data_dir=args.sisfall_dir, file_name="SisFall_adl_12.pth").down_sample(args.sisfall_down)
        adl_25_raw = GetPthData(pth_data_dir=args.sisfall_dir, file_name="SisFall_adl_25.pth").down_sample(args.sisfall_down)
        adl_100_raw = GetPthData(pth_data_dir=args.sisfall_dir, file_name="SisFall_adl_100.pth").down_sample(args.sisfall_down)
        fall_15_raw = GetPthData(pth_data_dir=args.sisfall_dir, file_name="SisFall_fall_15.pth").down_sample(args.sisfall_down)
        # adl_12_raw = GetPthData(pth_data_dir=args.sisfall_dir, file_name="SisFall_adl_12.pth").get_raw()
        # adl_25_raw = GetPthData(pth_data_dir=args.sisfall_dir, file_name="SisFall_adl_25.pth").get_raw()
        # adl_100_raw = GetPthData(pth_data_dir=args.sisfall_dir, file_name="SisFall_adl_100.pth").get_raw()
        # fall_15_raw = GetPthData(pth_data_dir=args.sisfall_dir, file_name="SisFall_fall_15.pth").get_raw()
        print(adl_12_raw.shape)
        print(adl_25_raw.shape)
        print(adl_100_raw.shape)
        print(fall_15_raw.shape)

        adl_12_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_12_raw, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.zeros(adl_12_fea_window.shape[0], dtype=torch.long)

        adl_25_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_25_raw, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_25_fea_window.shape[0], dtype=torch.long)])

        adl_100_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_100_raw, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_100_fea_window.shape[0], dtype=torch.long)])

        fall_15_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_15_raw, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_15_fea_window.shape[0], dtype=torch.long)])

        self.signals = torch.cat([adl_12_fea_window, adl_25_fea_window, adl_100_fea_window, fall_15_fea_window], dim=0)
        butterWorthFilter = ButterWorth(self.signals)
        self.signals_lowPass = torch.from_numpy(butterWorthFilter.lowPass()).to(torch.float32)
        self.signals = AD2Raw(ori_data=self.signals, ori_name='sisfall').fromSisfall()
        self.signals_lowPass = AD2Raw(ori_data=self.signals_lowPass, ori_name='sisfall').fromSisfall()  # 可以使用转换后的数据滤波，节省时间

        if need_pe:
            self.signals = PositionalEncoding(d_model=args.d_pe)(self.signals)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        signal_filtered = self.signals_lowPass[idx]
        label = self.label[idx]
        return signal, signal_filtered, label
