import numpy as np
import torch
import os


def txt_csv_files(path, file_roots, dir_names):
    lsdir = os.listdir(path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
    files = [i for i in lsdir if os.path.isfile(os.path.join(path, i))]
    if files:
        for f in files:
            file_roots.append(os.path.join(path, f))
    if dirs:
        for d in dirs:
            dir_names.append(d)
            txt_csv_files(os.path.join(path, d), file_roots, dir_names)

def see_sigle_file_shape(file_root):
    file = np.loadtxt(fname=file_root, delimiter=',', dtype=str)
    print(file_root, ":    ", file.shape)

def see_all_files_shape(file_roots):
    for i in range(len(file_roots)):
        see_sigle_file_shape(file_roots[i])

def overwrite(zeros,idx,data):
    zeros[idx, :data.shape[0], :] = data
def save_softfall2pth(softfall_file_roots):
    """
    Number Code Activity Trials Duration
    0 None Basic Fall 3 10s
    1 10204 蹲下侧向跌倒 3 15s
    2 10304 蹲下后向跌倒 3 15s
    3 10184 屈膝起立前向跌倒 3 15s
    4 10384 屈膝起立后向跌倒 3 15s
    5 10103 弯腰前向跌倒 3 15s
    6 10203 弯腰侧向跌倒 3 15s
    7 10383 弯腰起立后向跌倒 3 15s
    8 10075 躺下翻身竖向跌倒 3 15s
    9 11301 行走时后向滑倒 3 15s
    10 12101 行走时前向绊倒 3 15s
    11 13201 行走时侧向撞倒 3 15s
    12 14201 行走时直接侧向昏倒 3 15s
    13 12102 跑步时前向绊倒 3 15s
    14 11202 跑步时侧向滑倒 3 15s
    """
    fall_basic = torch.zeros(134, 1600, 9)
    fall_10204 = torch.zeros(34,1600,9)
    fall_10304 = torch.zeros(33,1600,9)
    fall_10184 = torch.zeros(33,1600,9)
    fall_10384 = torch.zeros(32,1600,9)
    fall_10103 = torch.zeros(33,1600,9)
    fall_10203 = torch.zeros(33,1600,9)
    fall_10383 = torch.zeros(33,1600,9)
    fall_10075 = torch.zeros(32,1600,9)
    fall_11301 = torch.zeros(34,1600,9)
    fall_12101 = torch.zeros(33,1600,9)
    fall_13201 = torch.zeros(34,1600,9)
    fall_14201 = torch.zeros(34,1600,9)
    fall_12102 = torch.zeros(33,1600,9)
    fall_11202 = torch.zeros(34,1600,9)


    """
    Number Code Activity Trials Duration
    0 None Flow ADL 1 主动记录各时间节点
    1 21000 (慢)走 1 100s
    2 22000 (慢)跑 1 100s
    3 23000 捡东西/弯腰 3 15s
    4 29000 上、下楼 3 15s
    5 25480 躺-坐-站 3 15s
    6 28450 站-坐-躺 3 15s
    """
    adl_flow = torch.zeros(12,1600,9)
    adl_21000 = torch.zeros(12,1600,9)
    adl_22000 = torch.zeros(12,1600,9)
    adl_23000 = torch.zeros(33,1600,9)
    adl_29000_up = torch.zeros(34,1600,9)
    adl_29000_down = torch.zeros(34,1600,9)
    adl_25480 = torch.zeros(33,1600,9)
    adl_28450 = torch.zeros(33,1600,9)


    idx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # idx = torch.zeros(20)
    for i in range(len(softfall_file_roots)):
        motion = softfall_file_roots[i].split('\\')[-1].split('_')[1]
        print(softfall_file_roots[i], motion, i)
        data = torch.from_numpy(np.loadtxt(fname=softfall_file_roots[i], delimiter=','))

        # annotated = np.loadtxt(fname=mobiact_file_roots[i], dtype=str, delimiter=',', skiprows=1, usecols=11)
        if motion == '10204':
            overwrite(fall_10204,idx[0], data)
            # print(BSC_Fall[idx[0]])
            idx[0] += 1
        elif motion == '10304':
            overwrite(fall_10304, idx[1], data)
            idx[1] += 1
        elif motion == '10184':
            overwrite(fall_10184, idx[2], data)
            idx[2] += 1
        elif motion == '10384':
            overwrite(fall_10384, idx[3], data)
            idx[3] += 1
        elif motion == '10103':
            overwrite(fall_10103, idx[4], data)
            idx[4] += 1
        elif motion == '10203':
            overwrite(fall_10203, idx[5], data)
            idx[5] += 1
        elif motion == '10383':
            overwrite(fall_10383, idx[6], data)
            idx[6] += 1
        elif motion == '10075':
            overwrite(fall_10075, idx[7], data)
            idx[7] += 1
        elif motion == '11301':
            overwrite(fall_11301, idx[8], data)
            idx[8] += 1
        elif motion == '12101':
            overwrite(fall_12101, idx[9], data)
            idx[9] += 1
        elif motion == '13201':
            overwrite(fall_13201, idx[10], data)
            idx[10] += 1
        elif motion == '14201':
            overwrite(fall_14201, idx[11], data)
            idx[11] += 1
        elif motion == '12102':
            overwrite(fall_12102, idx[12], data)
            idx[12] += 1
        elif motion == '11202':
            overwrite(fall_11202, idx[13], data)
            idx[13] += 1
        elif motion == 'basicfront' or motion == 'basicback' or motion == 'basicleft' or motion == 'basicright':
            overwrite(fall_basic, idx[14], data)
            idx[14] += 1
        elif motion == 'flowadl':
            overwrite(adl_flow, idx[15], data)
            idx[15] += 1
        elif motion == '21000':
            overwrite(adl_21000, idx[16], data)
            idx[16] += 1
        elif motion == '22000':
            overwrite(adl_22000, idx[17], data)
            idx[17] += 1
        elif motion == '23000':
            overwrite(adl_23000, idx[18], data)
            idx[18] += 1
        elif motion == '29000_up':
            overwrite(adl_29000_up, idx[19], data)
            idx[19] += 1
        elif motion == '29000_down':
            overwrite(adl_29000_down, idx[20], data)
            idx[20] += 1
        elif motion == '25480':
            overwrite(adl_25480, idx[21], data)
            idx[21] += 1
        elif motion == '28450':
            overwrite(adl_28450, idx[22], data)
            idx[22] += 1

    torch.save(fall_basic, "fall_basic.pth")
    torch.save(fall_10204, "fall_10204.pth")
    torch.save(fall_10304, "fall_10304.pth")
    torch.save(fall_10184, "fall_10184.pth")
    torch.save(fall_10384, "fall_10384.pth")
    torch.save(fall_10103, "fall_10103.pth")
    torch.save(fall_10203, "fall_10203.pth")
    torch.save(fall_10383, "fall_10383.pth")
    torch.save(fall_10075, "fall_10075.pth")
    torch.save(fall_11301, "fall_11301.pth")
    torch.save(fall_12101, "fall_12101.pth")
    torch.save(fall_13201, "fall_13201.pth")
    torch.save(fall_14201, "fall_14201.pth")
    torch.save(fall_12102, "fall_12102.pth")
    torch.save(fall_11202, "fall_11202.pth")
    torch.save(adl_flow, "adl_flow.pth")
    torch.save(adl_21000, "adl_21000.pth")
    torch.save(adl_22000, "adl_22000.pth")
    torch.save(adl_23000, "adl_23000.pth")
    torch.save(adl_29000_up, "adl_29000_up.pth")
    torch.save(adl_29000_down, "adl_29000_down.pth")
    torch.save(adl_25480, "adl_25480.pth")
    torch.save(adl_28450, "adl_28450.pth")

# softfall_path = r'C:\Users\Administrator\Desktop\NoBurrandNaNCSV'
# softfall_file_roots = []
# softfall_dir_names = []
# txt_csv_files(softfall_path, softfall_file_roots, softfall_dir_names)
# save_softfall2pth(softfall_file_roots)


# b = torch.cat([torch.ones(9,9),torch.zeros(4,9)],dim=0)
# print(b)
# print(b.shape)
#
# non_zero_rows = torch.abs(b).sum(dim=-1) > 0
# b = b[non_zero_rows]
# print(b)
# print(b.shape)


# x = torch.load('fall_10075.pth')
# print(x.shape)
# x = torch.transpose(x,1,2)
# print(x.shape)
#
# from utils import drawAxisPicturesWithData
#
# draw = drawAxisPicturesWithData(batch_data=x[0:4,...], picture_title='temp')
# draw.pltAllAxis()






