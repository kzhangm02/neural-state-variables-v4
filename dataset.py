import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, TensorDataset

class NeuralPhysDataset(Dataset):
    def __init__(self, data_filepath, num_frames, flag, seed, object_name=None):
        self.data_filepath = data_filepath
        self.num_frames = num_frames
        self.flag = flag
        self.seed = seed
        self.object_name = object_name
        self.all_filelist = self.get_all_filelist()

    def get_all_filelist(self):
        filelist = []
        obj_filepath = os.path.join(self.data_filepath, self.object_name)
        with open(os.path.join('../datainfo', self.object_name, f'data_split_dict_{self.seed}.json'), 'r') as file:
            seq_dict = json.load(file)
        vid_list = seq_dict[self.flag]

        # go through all the selected videos and get the triplets: input(t, t+1), output(t+2)
        for vid_idx in vid_list:
            seq_filepath = os.path.join(obj_filepath, str(vid_idx))
            num_frames = len(os.listdir(seq_filepath))
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            for p_frame in range(num_frames - 3):
                par_list = []
                for p in range(4):
                    par_list.append(os.path.join(seq_filepath, str(p_frame + p) + '.' + suf))
                filelist.append(par_list)
        return filelist

    def __len__(self):
        return len(self.all_filelist) // (self.num_frames - 3)

    def __getitem__(self, idx):
        start = (self.num_frames - 3) * idx
        end = (self.num_frames - 3) * (idx+1)
        file_list = self.all_filelist[start : end]
        data, target, filepath = [], [], []
        for par_list in file_list:
            tmp_data, tmp_target = [], []
            for i in range(2):
                tmp_data.append(self.get_data(par_list[i])) # 0, 1
            tmp_data = torch.cat(tmp_data, 2)
            data.append(tmp_data)
            tmp_target.append(self.get_data(par_list[-2])) # 2
            tmp_target.append(self.get_data(par_list[-1])) # 3
            tmp_target = torch.cat(tmp_target, 2)
            target.append(tmp_target)
            filepath.append('_'.join(par_list[0].split('/')[-2:]))
        data = torch.stack(data)
        target = torch.stack(target)
        return data, target, filepath

    def get_data(self, filepath):
        data = Image.open(filepath)
        data = data.resize((128, 128))
        data = np.array(data)
        data = torch.tensor(data / 255.0)
        data = data.permute(2, 0, 1).float()
        return data


class NeuralPhysRefineDataset(Dataset):
    def __init__(self, data, target, filepaths, num_frames):
        self.data = data
        self.target = target
        self.filepaths = filepaths
        self.num_frames = num_frames

    def __len__(self):
        return len(self.filepaths) // (self.num_frames - 3)

    def __getitem__(self, idx):
        start = (self.num_frames - 3) * idx
        end = (self.num_frames - 3) * (idx + 1)
        data = self.data[start : end]
        target = self.target[start : end]
        filepath = [f'{int(f[0])}_{int(f[1])}.png' for f in self.filepaths[start : end]]
        return data, target, filepath