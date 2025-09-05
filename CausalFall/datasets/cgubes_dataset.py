from paras import get_parameters
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from features import GetFeaWinFromDataset
from utils import GetPthData, ButterWorth, PositionalEncoding, modalEmbeddingAblation, modalEmbedding


class MyCGUBESData(Dataset):
    def __init__(self, need_pe):
        super(MyCGUBESData, self).__init__()
        adl_raw = GetPthData(pth_data_dir=args.cgubes_dir, file_name="cgubes_adl.pth").down_sample(args.cgubes_down)
        fall_raw = GetPthData(pth_data_dir=args.cgubes_dir, file_name="cgubes_fall.pth").down_sample(args.cgubes_down)
        print(adl_raw.shape)
        print(fall_raw.shape)
        adl_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_raw, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.zeros(adl_fea_window.shape[0], dtype=torch.long)

        fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_raw, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_fea_window.shape[0], dtype=torch.long)])

        self.signals = torch.cat([adl_fea_window, fall_fea_window], dim=0)

        butterWorthFilter = ButterWorth(self.signals)

        self.signals_lowPass = torch.from_numpy(butterWorthFilter.lowPass()).to(torch.float32)

        if need_pe:
            self.signals = PositionalEncoding(d_model=args.d_pe)(self.signals)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        signal_filtered = self.signals_lowPass[idx]
        label = self.label[idx]
        return signal, signal_filtered, label