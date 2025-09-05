import os
import csv
import numpy as np
import torch
import time
import shutil

# https://sites.google.com/view/kfalldataset
# Under sensor_data.zip file, there are 32 sub folders named by subject ID (e.g., SA06).
# Under each sub folder, it includes all the motion files from the corresponding subject.
# The naming rule of the motion file is explained here.

# Take "SA06T01R01.csv" as an example
# "SA06" means Subject ID is 06;
# "T01" means Task ID is 01;   Fall: 20<=TaskID<=36; Other: ADL
# "R01" means Trial ID is 01.
# For each motion file (.csv), it contains 11 columns which are
# TimeStamp(s), FrameCounter, acceleration (unit: g); gyroscope (unit: °/s) and euler angle(°) along three axes.


def classfication2folder(path):
    for subject_item in os.listdir(path):
        print(subject_item)
        for csv_item in os.listdir(os.path.join(path, subject_item)):
            classname = csv_item.split("T")[1]
            classname = classname.split("R")[0]
            classname = int(classname)
            shutil.copy(os.path.join(path, subject_item, csv_item), all_dir)
            if 20 <= classname <= 34:
                shutil.copy(os.path.join(path, subject_item, csv_item), fall_dir)
            else:
                shutil.copy(os.path.join(path, subject_item, csv_item), adl_dir)


class GetTensorData:
    def fall_csv2tensor(self, fall_path):
        fall_file = np.empty((2346, 1500, 9), dtype=np.float32)
        fall_idx = 0
        for file in os.listdir(fall_path):
            # print(file)
            with open(os.path.join(fall_path, file), 'r') as f:
                csv_reader = csv.reader(f)
                line_idx = 0
                skip_index = 0
                for line in csv_reader:
                    if skip_index == 0:
                        skip_index += 1
                        continue
                    if line[0] == '':
                        continue
                    # print(line_idx)
                    print(line[2:])
                    fall_file[fall_idx, line_idx, :] = line[2:]
                    line_idx += 1
                    # break
                    if line_idx == 1500:
                        # print(fall[fall_idx].shape)
                        # print(fall[fall_idx])
                        break
            print('---------------------------------Fall: ', fall_idx, '---------------------------------------------')
            fall_idx += 1

        return torch.tensor(fall_file)  # [1798,3000,9]

    def adl_csv2tensor(self, adl_path):
        adl_file_1500 = np.empty((2601, 1500, 9), dtype=np.float32)
        adl_file_3500 = np.empty((128, 3500, 9), dtype=np.float32)

        adl_1500_path = f"{adl_path}/1500"
        adl_3500_path = f"{adl_path}/3500"

        adl_idx = [0, 0]
        for file in os.listdir(adl_1500_path):
            with open(os.path.join(adl_1500_path, file), 'r') as f:
                csv_reader = csv.reader(f)
                line_idx = 0
                skip_index = 0
                for line in csv_reader:
                    if skip_index == 0:
                        skip_index += 1
                        continue
                    if line[0] == '':
                        continue
                    adl_file_1500[adl_idx[0], line_idx, :] = line[2:]
                    line_idx += 1
                    if line_idx == 1500:
                        break
            adl_idx[0] += 1
            print('---------------------------------ADL_1500:', adl_idx[0],
                  '---------------------------------------------')

        for file in os.listdir(adl_3500_path):
            with open(os.path.join(adl_3500_path, file), 'r') as f:
                csv_reader = csv.reader(f)
                line_idx = 0
                skip_index = 0
                for line in csv_reader:
                    if skip_index == 0:
                        skip_index += 1
                        continue
                    if line[0] == '':
                        continue
                    adl_file_3500[adl_idx[1], line_idx, :] = line[2:]
                    line_idx += 1
                    if line_idx == 3500:
                        break
            adl_idx[1] += 1
            print('---------------------------------ADL_3500:', adl_idx[1],
                  '---------------------------------------------')

        return torch.tensor(adl_file_1500), torch.tensor(adl_file_3500)  # [2060,2400,9] [492,5000,9] [150,20000,9]


if __name__ == '__main__':
    main_dir = r""

    raw_data_path = f"{main_dir}/sensor_data"

    fall_dir = f"{main_dir}/Fall"
    if not os.path.exists(fall_dir):
        print(fall_dir)
        os.makedirs(fall_dir)

    adl_dir = f"{main_dir}/ADL"
    if not os.path.exists(adl_dir):
        print(adl_dir)
        os.makedirs(adl_dir)

    all_dir = f"{main_dir}/ALL"
    if not os.path.exists(all_dir):
        print(all_dir)
        os.makedirs(all_dir)

    GTD = GetTensorData()
    fall = GTD.fall_csv2tensor(fall_dir)
    adl_1500, adl_3500 = GTD.adl_csv2tensor(adl_dir)
    print(fall.shape)
    print(adl_1500.shape, adl_3500.shape)
    torch.save(adl_1500, "KFall_ADL_1500.pth")
    torch.save(adl_3500, "KFall_ADL_3500.pth")
    torch.save(fall, "KFall_Fall.pth")
