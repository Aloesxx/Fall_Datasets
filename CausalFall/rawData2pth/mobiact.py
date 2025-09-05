import numpy as np
import torch
import os

"""
Activities of Daily Living:
+----+-------+----------------------------+--------+----------+---------------------------------------------------+
| No.| Label | Activity                   | Trials | Duration | Description                                       |
+----+-------+----------------------------+--------+----------+---------------------------------------------------+
| 1  | STD   | Standing                   | 1      | 5min     | Standing with subtle movements                    |
| 2  | WAL   | Walking                    | 1      | 5min     | Normal walking                                    |
| 3  | JOG   | Jogging                    | 3      | 30s      | Jogging                                           |
| 4  | JUM   | Jumping                    | 3      | 30s      | Continuous jumping                                |
| 5  | STU   | Stairs up                  | 6      | 10s      | Stairs up (10 stairs)                             |
| 6  | STN   | Stairs down                | 6      | 10s      | Stairs down (10 stairs)                           |
| 7  | SCH   | Stand to sit(sit on chair) | 6      | 6s       | Transition from standing to sitting               |
| 8  | SIT   | Sitting on chair           | 1      | 1min     | Sitting on a chair with subtle movements          |
| 9  | CHU   | Sit to stand(chair up)     | 6      | 6s       | Transition from sitting to standing               |
| 10 | CSI   | Car-step in                | 6      | 6s       | Step in a car                                     |
| 11 | CSO   | Car-step out               | 6      | 6s       | Step out a car                                    |
| 12 | LYI   | Lying                      | 12     | -        | Activity taken from the lying period after a fall |
+----+-------+----------------------------+--------+----------+---------------------------------------------------+


Falls:
+----+-------+--------------------+--------+----------+---------------------------------------------------------+
| No.| Label | Activity           | Trials | Duration | Description                                             |
+----+-------+--------------------+--------+----------+---------------------------------------------------------+
| 10 | FOL   | Forward-lying      | 3      | 10s      | Fall Forward from standing, use of hands to dampen fall |
| 11 | FKL   | Front-knees-lying  | 3      | 10s      | Fall forward from standing, first impact on knees       |
| 12 | BSC   | Back-sitting-chair | 3      | 10s      | Fall backward while trying to sit on a chair            |
| 13 | SDL   | Sideward-lying     | 3      | 10s      | Fall sidewards from standing, bending legs              |
+----+-------+--------------------+--------+----------+---------------------------------------------------------+

Scenarios:
+---------------------------------------------------------------------------------------------+
| 1st Scenario of Leaving the Home (SLH), Total duration 125拻 at max (1 trial/participant)   |
+----+-------+--------------------+-----------------------------------------------------------+
| No.| Label | Activity           | Description                                               |
+----+-------+--------------------+-----------------------------------------------------------+
| 1  | STD   | Standing           |  The recording starts with the participant standing       |
| 2  | WAL   | Walking            |  outside the door and locking the door. Then walks        |
| 3  | STN   | Stairs down        |  and descent stairs to leave his home. Following, he      |
| 4  | WAL   | Walking            |  riches the parking area where he stands in front of the  |
| 5  | STD   | Standing           |  car, unlocks the lock of the car, opens the door and     |
| 6  | CSI   | Car-step in        |  gets in the car. He remains sited for some seconds,      |
| 7  | SIT   | Sitting on chair   |  then he gets out of the car, closes the door and stands  |
| 8  | CSO   | Car-step out       |  in front of the door to lock the car.                    |
| 9  | STD   | Standing           |                                                           |
+-----+------+--------------------+-----------------------------------------------------------+
| 2nd Scenario of Being at work (SBW), Total duration 185拻 at max (1 trial/participant)      |
+----+-------+--------------------+-----------------------------------------------------------+
| No.| Label | Activity           | Description                                               |
+----+-------+--------------------+-----------------------------------------------------------+
| 1  | STD   | Standing           |  The recording starts with the participant standing       |
| 2  | WAL   | Walking            |  outside the cars door. Then walks from the parking       |
| 3  | STU   | Stairs up          |  area to his work building. He walks and ascent stairs    |
| 4  | WAL   | Walking            |  till he riches his office where he stops in front of the |
| 5  | STD   | Standing           |  door. Once he finds the keys he opens the door, gets     |
| 6  | WAL   | Walking            |  in his office and walks to his chair, where he sits.     |
| 7  | SCH   | Stand to sit       |                                                           |
| 8  | SIT   | Sitting on chair   |                                                           |
+-----+------+--------------------+-----------------------------------------------------------+
| 3rd Scenario of Leaving work (SLW), Total duration 185拻 at max (1 trial/participant)       |
+----+-------+--------------------+-----------------------------------------------------------+
| No.| Label | Activity           | Description                                               |
+----+-------+--------------------+-----------------------------------------------------------+
| 1  | SIT   | Sitting on chair   |  The recording starts with the participant sitting in the |
| 2  | CHU   | Sit to stand       |  chair in his office area. Then he gets up from the       |
| 3  | WAL   | Walking            |  chair, walks to the door and stands outside the office   |
| 4  | STD   | Standing           |  door. Once he find the keys, he lock the door and        |
| 5  | WAL   | Walking            |  walks and descent stairs till he riches the parking      |
| 6  | STN   | Stairs down        |  area. He walks to his car and stands in front of the     |
| 7  | WAL   | Walking            |  car, unlocks the lock of the car, opens the door and     |
| 8  | STD   | Standing           |  gets in the car. He remains sited for some seconds,      |
| 9  | CSI   | Car-step in        |  then he gets out of the car, closes the door and stands  |
| 10 | SIT   | Sitting on chair   |  in front of the door to lock the car.                    |
| 11 | CSO   | Car-step out       |                                                           |
| 12 | STD   | Standing           |                                                           |
+----+-------+--------------------+-----------------------------------------------------------+
| 4th Scenario of Being Exercise (SBE), Total duration 125拻 at max (1 trial/participant)     |
+----+-------+--------------------+-----------------------------------------------------------+
| No.| Label | Activity           | Description                                               |
+----+-------+--------------------+-----------------------------------------------------------+
| 1  | STD   | Standing           |  The recording starts with the participant standing in    |
| 2  | WAL   | Walking            |  front of the car. He starts his exercise by walking,     |
| 3  | JOG   | Jogging            |  then starts jogging from some seconds and once again     |
| 4  | WAL   | Walking            |  walking. Then he stops for some seconds to get a         |
| 5  | STD   | Standing           |  breath and he starts jumping and once more he            |
| 6  | JUM   | Jumping            |  standing to relax a little. Finally he walks till his    |
| 7  | STD   | Standing           |  car and stands outside the door.                         |
| 8  | WAL   | Walking            |                                                           |
| 9  | STD   | Standing           |                                                           |
+----+-------+--------------------+-----------------------------------------------------------+
| 5th Scenario of Returning at Home (SRH), Total duration 155拻 at max (1 trial/participant)  |
+----+-------+--------------------+-----------------------------------------------------------+
| No.| Label | Activity           | Description                                               |
+----+-------+--------------------+-----------------------------------------------------------+
| 1  | STD   | Standing           |  The recording starts with the participant standing       |
| 2  | CSI   | Car-step in        |  outside the cars door. He unlocks the lock of the car,   |
| 3  | SIT   | Sitting on chair   |  opens the door and gets in the car. He remains sited     |
| 4  | CSO   | Car-step out       |  for some seconds, then he gets out of the car, closes    |
| 5  | STD   | Standing           |  the door and stands in front of the door to lock the     |
| 6  | WAL   | Walking            |  car.  Then walks from the parking area to his home.      |
| 7  | STU   | Stairs up          |  He walks and ascent stairs till riches his home door,    |
| 8  |  WAL  | Walking            |  where he stands to finds the keys. Then he opens the     |
| 9  | STD   | Standing           |  door, gets in his home, walks till a chair and sits.     |
| 10 | WAL   | Walking            |                                                           |
| 11 | SCH   | Stand to sit       |                                                           |
+----+-------+--------------------+-----------------------------------------------------------+



timestamp	rel_time	acc_x	acc_y	acc_z	gyro_x	gyro_y	gyro_z	azimuth	pitch	roll	label
"""
# data = np.loadtxt(fname=filename, delimiter=',', skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10))
# annotated = np.loadtxt(fname=filename, dtype=str, delimiter=',', skiprows=1, usecols=11)
# print(data.shape)
# print(annotated.shape)


def txt_csv_files(path, file_roots, dir_names):
    lsdir = os.listdir(path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
    files = [i for i in lsdir if os.path.isfile(os.path.join(path, i))]
    if files:
        for f in files:
            # print(os.path.join(path, f))
            file_roots.append(os.path.join(path, f))
    if dirs:
        for d in dirs:
            dir_names.append(d)
            txt_csv_files(os.path.join(path, d), file_roots, dir_names)  # 递归查找

def see_sigle_file_shape(file_root):
    file = np.loadtxt(fname=file_root, delimiter=',', dtype=str)
    print(file_root, ":    ", file.shape)

def see_all_files_shape(file_roots):
    for i in range(len(file_roots)):
        see_sigle_file_shape(file_roots[i])

def overwrite(zeros,idx,data):
    zeros[idx, :data.shape[0], :] = data
def save_mobiact2pth(mobiact_file_roots):
    BSC_Fall = torch.zeros(191,2000,11)
    FKL_Fall = torch.zeros(192,2000,11)
    FOL_Fall = torch.zeros(192,2000,11)
    SDL_Fall = torch.zeros(192,2000,11)
    CHU_ADL = torch.zeros(114,1200,11)
    CSI_ADL = torch.zeros(358,1200,11)
    CSO_ADL = torch.zeros(360,1200,11)
    JOG_ADL = torch.zeros(183,6000,11)
    JUM_ADL = torch.zeros(183,6000,11)
    SCH_ADL = torch.zeros(365, 1200, 11)
    SIT_ADL = torch.zeros(19, 12000, 11)
    STD_ADL = torch.zeros(60,60000,11)
    STN_ADL = torch.zeros(365,2000,11)
    STU_ADL = torch.zeros(364,2000,11)
    WAL_ADL = torch.zeros(61,60000,11)
    # flow adl
    SLH_Flow_ADL = torch.zeros(19,24000,11)      # Scenario 1
    SBW_Flow_ADL = torch.zeros(19, 36000, 11)    # Scenario 2
    SLW_Flow_ADL = torch.zeros(19,36000,11)      # Scenario 3
    SBE_Flow_ADL = torch.zeros(19, 36000, 11)    # Scenario 4
    SRH_Flow_ADL = torch.zeros(19,30000,11)      # Scenario 5

    idx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # idx = torch.zeros(20)
    for i in range(len(mobiact_file_roots)):
        motion = mobiact_file_roots[i].split('\\')[-1].split('_')[0]
        data = torch.from_numpy(np.loadtxt(fname=mobiact_file_roots[i], delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)))

        if motion == 'BSC':
            overwrite(BSC_Fall,idx[0], data)
            idx[0] += 1
        elif motion == 'FKL':
            overwrite(FKL_Fall, idx[1], data)
            idx[1] += 1
        elif motion == 'FOL':
            overwrite(FOL_Fall, idx[2], data)
            idx[2] += 1
        elif motion == 'SDL':
            overwrite(SDL_Fall, idx[3], data)
            idx[3] += 1
        elif motion == 'CHU':
            overwrite(CHU_ADL, idx[4], data)
            idx[4] += 1
        elif motion == 'CSI':
            overwrite(CSI_ADL, idx[5], data)
            idx[5] += 1
        elif motion == 'CSO':
            overwrite(CSO_ADL, idx[6], data)
            idx[6] += 1
        elif motion == 'JOG':
            overwrite(JOG_ADL, idx[7], data)
            idx[7] += 1
        elif motion == 'JUM':
            overwrite(JUM_ADL, idx[8], data)
            idx[8] += 1
        elif motion == 'SCH':
            overwrite(SCH_ADL, idx[9], data)
            idx[9] += 1
        elif motion == 'SIT':
            overwrite(SIT_ADL, idx[10], data)
            idx[10] += 1
        elif motion == 'STD':
            overwrite(STD_ADL, idx[11], data)
            idx[11] += 1
        elif motion == 'STN':
            overwrite(STN_ADL, idx[12], data)
            idx[12] += 1
        elif motion == 'STU':
            overwrite(STU_ADL, idx[13], data)
            idx[13] += 1
        elif motion == 'WAL':
            overwrite(WAL_ADL, idx[14], data)
            idx[14] += 1
        elif motion == 'SLH':
            overwrite(SLH_Flow_ADL, idx[15], data)
            idx[15] += 1
        elif motion == 'SBW':
            overwrite(SBW_Flow_ADL, idx[16], data)
            idx[16] += 1
        elif motion == 'SLW':
            overwrite(SLW_Flow_ADL, idx[17], data)
            idx[17] += 1
        elif motion == 'SBE':
            overwrite(SBE_Flow_ADL, idx[18], data)
            idx[18] += 1
        elif motion == 'SRH':
            overwrite(SRH_Flow_ADL, idx[19], data)
            idx[19] += 1

    torch.save(BSC_Fall, "BSC_Fall.pth")
    torch.save(FKL_Fall, "FKL_Fall.pth")
    torch.save(FOL_Fall, "FOL_Fall.pth")
    torch.save(SDL_Fall, "SDL_Fall.pth")
    torch.save(CHU_ADL, "CHU_ADL.pth")
    torch.save(CSI_ADL, "CSI_ADL.pth")
    torch.save(CSO_ADL, "CSO_ADL.pth")
    torch.save(JOG_ADL, "JOG_ADL.pth")
    torch.save(JUM_ADL, "JUM_ADL.pth")
    torch.save(SCH_ADL, "SCH_ADL.pth")
    torch.save(SIT_ADL, "SIT_ADL.pth")
    torch.save(STD_ADL, "STD_ADL.pth")
    torch.save(STN_ADL, "STN_ADL.pth")
    torch.save(STU_ADL, "STU_ADL.pth")
    torch.save(WAL_ADL, "WAL_ADL.pth")
    torch.save(SBW_Flow_ADL, "SBW_Flow_ADL.pth")
    torch.save(SLH_Flow_ADL, "SLH_Flow_ADL.pth")
    torch.save(SLW_Flow_ADL, "SLW_Flow_ADL.pth")
    torch.save(SBE_Flow_ADL, "SBE_Flow_ADL.pth")
    torch.save(SRH_Flow_ADL, "SRH_Flow_ADL.pth")

mobiact_path = r''
mobiact_file_roots = []
mobiact_dir_names = []
txt_csv_files(mobiact_path, mobiact_file_roots, mobiact_dir_names)
save_mobiact2pth(mobiact_file_roots)
