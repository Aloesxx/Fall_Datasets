from paras import get_parameters
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from features import GetFeaWinFromDataset
from utils import GetPthData, ButterWorth, PositionalEncoding, modalEmbeddingAblation, modalEmbedding

_use_new_zipfile_serialization = False


class MyMobiactData(Dataset):
    def __init__(self, need_pe):
        super(MyMobiactData, self).__init__()
        args = get_parameters()
        CHU_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="CHU_ADL.pth").get_raw()
        CSI_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="CSI_ADL.pth").get_raw()
        CSO_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="CSO_ADL.pth").get_raw()
        JOG_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="JOG_ADL.pth").get_raw()
        # JUM_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="JUM_ADL.pth").get_raw()
        SBE_Flow_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SBE_Flow_ADL.pth").get_raw()
        SBW_Flow_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SBW_Flow_ADL.pth").get_raw()
        SCH_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SCH_ADL.pth").get_raw()
        SIT_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SIT_ADL.pth").get_raw()
        SLH_Flow_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SLH_Flow_ADL.pth").get_raw()
        SLW_Flow_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SLW_Flow_ADL.pth").get_raw()
        SRH_Flow_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SRH_Flow_ADL.pth").get_raw()
        STD_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="STD_ADL.pth").get_raw()
        STN_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="STN_ADL.pth").get_raw()
        STU_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="STU_ADL.pth").get_raw()
        WAL_ADL = GetPthData(pth_data_dir=args.mobiact_dir, file_name="WAL_ADL.pth").get_raw()
        BSC_Fall = GetPthData(pth_data_dir=args.mobiact_dir, file_name="BSC_Fall.pth").get_raw()
        # FKL_Fall = GetPthData(pth_data_dir=args.mobiact_dir, file_name="FKL_Fall.pth").get_raw()
        # FOL_Fall = GetPthData(pth_data_dir=args.mobiact_dir, file_name="FOL_Fall.pth").get_raw()
        SDL_Fall = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SDL_Fall.pth").get_raw()
        print(CHU_ADL.shape)
        print(SBE_Flow_ADL.shape)
        print(BSC_Fall.shape)

        # adl
        CHU_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=CHU_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.zeros(CHU_ADL_fea_window.shape[0], dtype=torch.long)
        CSI_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=CSI_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(CSI_ADL_fea_window.shape[0], dtype=torch.long)])
        CSO_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=CSO_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(CSO_ADL_fea_window.shape[0], dtype=torch.long)])
        JOG_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=JOG_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(JOG_ADL_fea_window.shape[0], dtype=torch.long)])
        # JUM_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=JUM_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        # self.label = torch.cat([self.label, torch.zeros(JUM_ADL_fea_window.shape[0], dtype=torch.long)])
        SBE_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SBE_Flow_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SBE_Flow_ADL_fea_window.shape[0], dtype=torch.long)])
        SBW_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SBW_Flow_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SBW_Flow_ADL_fea_window.shape[0], dtype=torch.long)])
        SCH_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SCH_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SCH_ADL_fea_window.shape[0], dtype=torch.long)])
        SIT_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SIT_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SIT_ADL_fea_window.shape[0], dtype=torch.long)])
        SLH_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SLH_Flow_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SLH_Flow_ADL_fea_window.shape[0], dtype=torch.long)])
        SLW_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SLW_Flow_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SLW_Flow_ADL_fea_window.shape[0], dtype=torch.long)])
        SRH_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SRH_Flow_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SRH_Flow_ADL_fea_window.shape[0], dtype=torch.long)])
        STD_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=STD_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(STD_ADL_fea_window.shape[0], dtype=torch.long)])
        STN_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=STN_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(STN_ADL_fea_window.shape[0], dtype=torch.long)])
        STU_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=STU_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(STU_ADL_fea_window.shape[0], dtype=torch.long)])
        WAL_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=WAL_ADL, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.zeros(WAL_ADL_fea_window.shape[0], dtype=torch.long)])
        # fall
        BSC_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=BSC_Fall, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(BSC_Fall_fea_window.shape[0], dtype=torch.long)])
        # FKL_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=FKL_Fall, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        # self.label = torch.cat([self.label, torch.ones(FKL_Fall_fea_window.shape[0], dtype=torch.long)])
        # FOL_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=FOL_Fall, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        # self.label = torch.cat([self.label, torch.ones(FOL_Fall_fea_window.shape[0], dtype=torch.long)])
        SDL_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SDL_Fall, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len).getWindow()
        self.label = torch.cat([self.label, torch.ones(SDL_Fall_fea_window.shape[0], dtype=torch.long)])

        self.signals = torch.cat([CHU_ADL_fea_window, CSI_ADL_fea_window, CSO_ADL_fea_window, JOG_ADL_fea_window,
                                  SBE_Flow_ADL_fea_window, SBW_Flow_ADL_fea_window,
                                  SCH_ADL_fea_window, SIT_ADL_fea_window, SLH_Flow_ADL_fea_window,
                                  SLW_Flow_ADL_fea_window, SRH_Flow_ADL_fea_window, STD_ADL_fea_window,
                                  STN_ADL_fea_window, STU_ADL_fea_window, WAL_ADL_fea_window, BSC_Fall_fea_window,
                                  SDL_Fall_fea_window], dim=0)
        self.signals = self.signals[:, 2:8, :]
        print(self.signals.shape)

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

