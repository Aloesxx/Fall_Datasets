from paras import get_parameters
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from features import GetFeaWinFromDataset
from utils import GetPthData, ButterWorth, PositionalEncoding, modalEmbeddingAblation, modalEmbedding


class MySlowfallData(Dataset):
    def __init__(self, need_pe):
        super(MySoftfallData, self).__init__()
        args = get_parameters()
        adl_21000_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="adl_21000.pth").get_raw()  # (慢)走
        adl_22000_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="adl_22000.pth").get_raw()  # (慢)跑
        adl_23000_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="adl_23000.pth").get_raw()  # 捡东西/弯腰
        adl_25480_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="adl_25480.pth").get_raw()  # 躺-坐-站
        adl_28450_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="adl_28450.pth").get_raw()  # 站-坐-躺
        adl_29000_up_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="adl_29000_up.pth").get_raw()  # 上楼
        adl_29000_down_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="adl_29000_down.pth").get_raw()  # 下楼
        adl_flow_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="adl_flow.pth").get_raw()  # ADL Flow
        fall_10075_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_10075.pth").get_raw()  # 躺下翻身竖向跌倒
        fall_10103_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_10103.pth").get_raw()  # 弯腰前向跌倒
        fall_10184_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_10184.pth").get_raw()  # 屈膝起立前向跌倒
        fall_10203_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_10203.pth").get_raw()  # 弯腰侧向跌倒
        fall_10204_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_10204.pth").get_raw()  # 蹲下侧向跌倒
        fall_10304_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_10304.pth").get_raw()  # 蹲下后向跌倒
        fall_10383_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_10383.pth").get_raw()  # 弯腰起立后向跌倒
        fall_10384_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_10384.pth").get_raw()  # 屈膝起立后向跌倒
        fall_11202_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_11202.pth").get_raw()  # 跑步时侧向滑倒
        fall_11301_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_11301.pth").get_raw()  # 行走时后向滑倒
        fall_12101_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_12101.pth").get_raw()  # 行走时前向绊倒
        fall_12102_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_12102.pth").get_raw()  # 跑步时前向绊倒
        fall_13201_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_13201.pth").get_raw()  # 行走时侧向撞倒
        fall_14201_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_14201.pth").get_raw()  # 行走时直接侧向昏倒
        fall_basic_raw = GetPthData(pth_data_dir=args.softfall_dir, file_name="fall_basic.pth").get_raw()  # Basic Fall

        adl_21000_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_21000_raw, pre_len=args.pre_len,
                                                    front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.zeros(adl_21000_fea_window.shape[0], dtype=torch.long)

        adl_22000_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_22000_raw, pre_len=args.pre_len,
                                                    front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_22000_fea_window.shape[0], dtype=torch.long)])

        adl_23000_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_23000_raw, pre_len=args.pre_len,
                                                    front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_23000_fea_window.shape[0], dtype=torch.long)])

        adl_25480_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_25480_raw, pre_len=args.pre_len,
                                                    front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_25480_fea_window.shape[0], dtype=torch.long)])

        adl_28450_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_28450_raw, pre_len=args.pre_len,
                                                    front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_28450_fea_window.shape[0], dtype=torch.long)])

        adl_29000_up_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_29000_up_raw, pre_len=args.pre_len,
                                                       front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_29000_up_fea_window.shape[0], dtype=torch.long)])

        adl_29000_down_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_29000_down_raw, pre_len=args.pre_len,
                                                         front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_29000_down_fea_window.shape[0], dtype=torch.long)])

        adl_flow_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_flow_raw, pre_len=args.pre_len,
                                                   front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_flow_fea_window.shape[0], dtype=torch.long)])

        fall_10075_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_10075_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_10075_fea_window.shape[0], dtype=torch.long)])

        fall_10103_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_10103_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_10103_fea_window.shape[0], dtype=torch.long)])

        fall_10184_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_10184_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_10184_fea_window.shape[0], dtype=torch.long)])

        fall_10203_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_10203_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_10203_fea_window.shape[0], dtype=torch.long)])

        fall_10204_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_10204_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_10204_fea_window.shape[0], dtype=torch.long)])

        fall_10304_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_10304_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_10304_fea_window.shape[0], dtype=torch.long)])

        fall_10383_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_10383_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_10383_fea_window.shape[0], dtype=torch.long)])

        fall_10384_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_10384_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_10384_fea_window.shape[0], dtype=torch.long)])

        fall_11202_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_11202_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_11202_fea_window.shape[0], dtype=torch.long)])

        fall_11301_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_11301_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_11301_fea_window.shape[0], dtype=torch.long)])

        fall_12101_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_12101_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_12101_fea_window.shape[0], dtype=torch.long)])

        fall_12102_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_12102_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_12102_fea_window.shape[0], dtype=torch.long)])

        fall_13201_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_13201_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_13201_fea_window.shape[0], dtype=torch.long)])

        fall_14201_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_14201_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_14201_fea_window.shape[0], dtype=torch.long)])

        fall_basic_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_basic_raw, pre_len=args.pre_len,
                                                     front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(fall_basic_fea_window.shape[0], dtype=torch.long)])

        self.signals = torch.cat(
            [adl_21000_fea_window, adl_22000_fea_window, adl_23000_fea_window, adl_25480_fea_window,
             adl_28450_fea_window, adl_29000_up_fea_window, adl_29000_down_fea_window,
             adl_flow_fea_window, fall_10075_fea_window, fall_10103_fea_window, fall_10184_fea_window,
             fall_10203_fea_window, fall_10204_fea_window,
             fall_10304_fea_window, fall_10383_fea_window, fall_10384_fea_window, fall_11202_fea_window,
             fall_11301_fea_window,
             fall_12101_fea_window, fall_12102_fea_window, fall_13201_fea_window, fall_14201_fea_window,
             fall_basic_fea_window], dim=0)

        butterWorthFilter = ButterWorth(self.signals)
        self.signals_lowPass = torch.from_numpy(butterWorthFilter.lowPass()).to(torch.float32)

        # PositionalEncoding
        if need_pe:
            self.signals = PositionalEncoding(d_model=args.d_pe)(self.signals)

        # self.signals = modalEmbedding(modal_num=args.modal_three, batch_data=self.signals)

        # self.signals = modalEmbeddingAblation(modal_num=args.modal_three, batch_data=self.signals)

        # self.signals_lowPass = PositionalEncoding(d_model=args.d_pe)(self.signals_lowPass)
        # self.signals_lowPass = modalEmbedding(modal_num=args.modal_three, batch_data=self.signals_lowPass)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        signal_filtered = self.signals_lowPass[idx]
        label = self.label[idx]

        return signal, signal_filtered, label
