from paras import get_parameters
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from features import GetFeaWinFromDataset
from utils import GetPthData, ButterWorth, PositionalEncoding, modalEmbeddingAblation, modalEmbedding


class MyKfallData(Dataset):
    def __init__(self, need_pe):
        super(MyKfallData, self).__init__()
        args = get_parameters()
        adl_1500_raw = GetPthData(pth_data_dir=args.kfall_dir, file_name="adl_1500.pth").down_sample(args.kfall_down)
        adl_3500_raw = GetPthData(pth_data_dir=args.kfall_dir, file_name="adl_3500.pth").down_sample(args.kfall_down)
        fall_raw = GetPthData(pth_data_dir=args.kfall_dir, file_name="fall.pth").down_sample(args.kfall_down)
        print(adl_1500_raw.shape)
        print(adl_3500_raw.shape)
        print(fall_raw.shape)

        adl_1500_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_1500_raw, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.zeros(adl_1500_fea_window.shape[0], dtype=torch.long)

        adl_3500_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_3500_raw, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_3500_fea_window.shape[0], dtype=torch.long)])

        fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_raw, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_fea_window.shape[0], dtype=torch.long)])

        
        self.signals = torch.cat([adl_1500_fea_window, adl_3500_fea_window, fall_fea_window], dim=0)
        print(self.signals.shape)

        butterWorthFilter = ButterWorth(self.signals)
        self.signals_lowPass = torch.from_numpy(butterWorthFilter.lowPass()).to(torch.float32)

        if need_pe:
            print(self.signals.shape)
            self.signals = PositionalEncoding(d_model=args.d_pe)(self.signals)
            print(self.signals.shape)

        # self.signals = modalEmbedding(modal_num=args.modal_three, batch_data=self.signals)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        signal_filtered = self.signals_lowPass[idx]
        label = self.label[idx]
        return signal, signal_filtered, label
