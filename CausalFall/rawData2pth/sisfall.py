import os
import csv
import numpy as np
import torch
import time


class GetTensorData():

    def fall_csv2tensor(self):

        fall_path = 'path/to/SisFall_dataset_csv/Fall'

        fall_file = np.empty((1798, 3000, 9), dtype=np.float32)
        fall_idx = 0
        for file in os.listdir(fall_path):
            with open(os.path.join(fall_path, file), 'r') as f:
                csv_reader = csv.reader(f)
                line_idx = 0
                for line in csv_reader:
                    if line[0] == '':
                        continue
                    fall_file[fall_idx, line_idx, :] = line[1:]
                    line_idx += 1
                    if line_idx == 3000:
                        break
            print('---------------------------------Fall: ', fall_idx, '---------------------------------------------')
            fall_idx += 1

        return torch.tensor(fall_file)

    def adl_csv2tensor(self):

        adl_path = 'path/to/SisFall_dataset_csv/ADL/'

        adl_file_12 = np.empty((2060, 2400, 9), dtype=np.float32)
        adl_file_25 = np.empty((492, 5000, 9), dtype=np.float32)
        adl_file_100 = np.empty((150, 20000, 9), dtype=np.float32)

        adl_idx = [0, 0, 0]
        for file in os.listdir(adl_path):
            d_num = int(file.split('_')[0].split('D')[1])
            if d_num < 5:  # 100s 150个 20000x9
                with open(os.path.join(adl_path, file), 'r') as f:
                    csv_reader = csv.reader(f)
                    line_idx = 0
                    for line in csv_reader:
                        if line[0] == '':
                            continue
                        adl_file_100[adl_idx[2], line_idx, :] = line[1:]
                        line_idx += 1
                        if line_idx == 20000:
                            break
                adl_idx[2] += 1
                print('---------------------------------ADL_D', d_num, ':', adl_idx[2],
                      '---------------------------------------------')
            elif d_num > 6 and d_num != 17:  # 12s 2060个 2400x9
                with open(os.path.join(adl_path, file), 'r') as f:
                    csv_reader = csv.reader(f)
                    line_idx = 0
                    for line in csv_reader:
                        if line[0] == '':
                            continue
                        adl_file_12[adl_idx[0], line_idx, :] = line[1:]
                        line_idx += 1
                        if line_idx == 2400:
                            break
                adl_idx[0] += 1
                print('---------------------------------ADL_D', d_num, ':', adl_idx[0],
                      '---------------------------------------------')

            else:  # 25s 492个5000x9
                with open(os.path.join(adl_path, file), 'r') as f:
                    csv_reader = csv.reader(f)
                    line_idx = 0
                    for line in csv_reader:
                        if line[0] == '':
                            continue
                        adl_file_25[adl_idx[1], line_idx, :] = line[1:]
                        line_idx += 1
                        if line_idx == 5000:
                            break
                adl_idx[1] += 1
                print('---------------------------------ADL_D', d_num, ':', adl_idx[1],
                      '---------------------------------------------')
        return torch.tensor(adl_file_12),torch.tensor(adl_file_25),torch.tensor(adl_file_100) # [2060,2400,9] [492,5000,9] [150,20000,9]

if __name__ == '__main__':
    GTD = GetTensorData()
    fall_15 = GTD.fall_csv2tensor()
    adl_12,adl_25,adl_100 = GTD.adl_csv2tensor()
    print(fall_15.shape)
    print(adl_12.shape,adl_25.shape,adl_100.shape)
    torch.save(adl_12,"SisFall_ADL_12.pth")
    torch.save(adl_25,"SisFall_ADL_25.pth")
    torch.save(adl_100,"SisFall_ADL_100.pth")
    torch.save(fall_15,"SisFall_Fall_15.pth")
