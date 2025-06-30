from paras import get_parameters
import torch
import os
import pandas as pd
import numpy as np
from features import GetFeaWinFromDataset
from utils import GetPthData
from fig_utils import draw_heatmap
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import grangercausalitytests
import torch.nn.functional as F

args = get_parameters()

def getGCT(axis_num, filepath, filename, down_step, Prelen, GC_len):
    GCT_Times = np.zeros([axis_num, axis_num])
    fileraw = GetPthData(pth_data_dir=filepath, file_name=filename).down_sample(down_step)
    file_fea_window = GetFeaWinFromDataset(
        multi_axis_dataset=fileraw,
        pre_len=Prelen,
        front_len=GC_len,
        rear_len=GC_len+Prelen).getWindow()
    for i in range(file_fea_window.shape[0]):
        for j in range(file_fea_window.shape[1]):
            df_pre = pd.DataFrame(file_fea_window[i, j, 0:GC_len])
            for k in range(file_fea_window.shape[1]):
                df_after = pd.DataFrame(
                    file_fea_window[i, k, GC_len+Prelen:GC_len*2+Prelen])
                df_concat = pd.concat([df_pre, df_after], axis=1)
                print(df_pre.shape)
                print(df_after.shape)
                print(df_concat.shape)
                print(filename)
                gct = grangercausalitytests(df_concat, maxlag=1)
                if (gct[1][0]["ssr_ftest"][1] < 0.05) and (gct[1][0]["ssr_chi2test"][1] < 0.05) and (
                        gct[1][0]["lrtest"][1] < 0.05) and (gct[1][0]["params_ftest"][1] < 0.05):
                    GCT_Times[j][k] = GCT_Times[j][k] + 1
    print(GCT_Times)
    return GCT_Times


if __name__ == '__main__':

    # Prelen = 15
    # GC_len = 38
    # cmp = "coolwarm"  # coolwarm hot
    # dn = "softfall"

    Prelen = 10
    GC_len = 70
    cmp = "hot"  # coolwarm hot
    dn = "slowfall"

    if dn == "slowfall":
        dataset_path = args.softfall_dir
        channel_type = "a,g,m"
        hz = 25
        down_step = 25 // hz

    # A, accelerometer; G, gyroscope; O, orientation measurement; M, magnetometer; B, barometer.
    channel_fallng = []
    channel_imp = []
    if "a" in channel_type:
        channel_fallng.append("Acc_X_Falling")
        channel_fallng.append("Acc_Y_Falling")
        channel_fallng.append("Acc_Z_Falling")
        channel_imp.append("Acc_X_Impact")
        channel_imp.append("Acc_Y_Impact")
        channel_imp.append("Acc_Z_Impact")
    if "g" in channel_type:
        channel_fallng.append("Gyro_X_Falling")
        channel_fallng.append("Gyro_Y_Falling")
        channel_fallng.append("Gyro_Z_Falling")
        channel_imp.append("Gyro_X_Impact")
        channel_imp.append("Gyro_Y_Impact")
        channel_imp.append("Gyro_Z_Impact")
    if "m" in channel_type:
        channel_fallng.append("Mag_X_Falling")
        channel_fallng.append("Mag_Y_Falling")
        channel_fallng.append("Mag_Z_Falling")
        channel_imp.append("Mag_X_Impact")
        channel_imp.append("Mag_Y_Impact")
        channel_imp.append("Mag_Z_Impact")
    if "o" in channel_type:
        channel_fallng.append("Ori_X_Falling")
        channel_fallng.append("Ori_Y_Falling")
        channel_fallng.append("Ori_Z_Falling")
        channel_imp.append("Ori_X_Impact")
        channel_imp.append("Ori_Y_Impact")
        channel_imp.append("Ori_Z_Impact")
    if dn == "sisfall":
        channel_fallng.append("Acc_X_Falling")
        channel_fallng.append("Acc_Y_Falling")
        channel_fallng.append("Acc_Z_Falling")
        channel_imp.append("Acc_X_Impact")
        channel_imp.append("Acc_Y_Impact")
        channel_imp.append("Acc_Z_Impact")
    print(channel_fallng)
    axis_num = len(channel_fallng)
    print(axis_num)
    channel_name = [channel_fallng, channel_imp]

    savePath = args.gcgraph_dir
    file_path = f"{savePath}/{dn}_{channel_type}_{hz}Hz_Prelen{Prelen}_GClen{GC_len}_GCGraph.csv"

    channel_mat = np.zeros([axis_num, axis_num])
    for dataFile in os.listdir(dataset_path):
        if "fall" in dataFile:
            if dn == "softfall":
                if dataFile == "fall_10103.pth" or dataFile == "fall_10204.pth" or dataFile == "fall_basic.pth":
                    continue
            print(dataFile)
            channel_mat = channel_mat + getGCT(axis_num, dataset_path, dataFile, down_step=down_step, Prelen=Prelen, GC_len=GC_len)
    print(torch.tensor(channel_mat))
    df_mat = pd.DataFrame(channel_mat)
    df_mat.to_csv(file_path)
    channel_weight = torch.sum(torch.tensor(channel_mat), axis=1)
    print(channel_weight)

    # # draw Heatmap
    # # channel_df = pd.read_csv(file_path, index_col=0)
    # isSave = True
    # draw_heatmap(channel_name=channel_name, channel_df=file_path, cmp=cmp, isSave=isSave,
    #              savePath=f"{savePath}/{dn}_{channel_type}_{hz}Hz_Prelen{Prelen}_GClen{GC_len}_{cmp}_heatmap_notitle_rotation.png")




