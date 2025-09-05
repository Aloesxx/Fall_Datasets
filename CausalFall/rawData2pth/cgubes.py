import os
import csv
import numpy as np
import torch
import time
import shutil

def classfication2folder(path):
    findstr = "Fall"
    for txt_item in os.listdir(path):
        if findstr in txt_item:
            shutil.copy(os.path.join(path, txt_item), fall_dir)
        else:
            shutil.copy(os.path.join(path, txt_item), adl_dir)


class GetTensorData:
    def fall_txt2tensor(self, fall_path):
        fall_file = np.empty((60, 3500, 6), dtype=np.float32)
        fall_idx = 0
        for file in os.listdir(fall_path):
            # print(file)
            with open(os.path.join(fall_path, file), 'r') as f:
                lines = f.readlines()
                line_idx = 0
                for line in lines[4:]:
                    # if line[0] == '':
                    #     continue
                    # print(line)  # '-0.084249, 0.873016, 0.531136, 0.234895, -0.026849, -0.060167\n'
                    line = line.strip().split(', ')
                    # print(line)  # ['-0.084249', '0.873016', '0.531136', '0.234895', '-0.026849', '-0.060167']
                    fall_file[fall_idx, line_idx, :] = line
                    line_idx += 1
                    # print(line[1:])
                    # break
                    if line_idx == 3500:
                        # print(fall[fall_idx].shape)
                        # print(fall[fall_idx])
                        break
            print('---------------------------------Fall: ', fall_idx, '---------------------------------------------')
            fall_idx += 1
        return torch.tensor(fall_file)

    def adl_txt2tensor(self, adl_path):
        # 区分是较长时间和较短时间的adl
        adl_file = np.empty((135, 3500, 6), dtype=np.float32)
        adl_idx = 0
        for file in os.listdir(adl_path):
            with open(os.path.join(adl_path, file), 'r') as f:
                lines = f.readlines()
                line_idx = 0
                for line in lines[4:]:
                    line = line.strip().split(', ')
                    adl_file[adl_idx, line_idx, :] = line
                    line_idx += 1
                    if line_idx == 3500:
                        break
            adl_idx += 1
            print('---------------------------------ADL:', adl_idx,
                  '---------------------------------------------')
        return torch.tensor(adl_file)


if __name__ == '__main__':
    main_dir = r""

    raw_data_path = f"{main_dir}/ALL"

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
    fall = GTD.fall_txt2tensor(fall_dir)
    adl = GTD.adl_txt2tensor(adl_dir)
    torch.save(fall, f"{main_dir}/cgubes_fall.pth")
    torch.save(adl, f"{main_dir}/cgubes_adl.pth")
