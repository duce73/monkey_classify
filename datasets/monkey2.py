import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile

import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class MONKEY2Dataset(Dataset):
    def __init__(self, root_dir, clip_len, split='1', train=True, transforms_=None, test_sample_num=10):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.train = train
        self.split = split
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path,header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None)[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print("use split" + self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        if self.train:
            clip_start = random.randint(0, length - self.clip_len)
            clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame)
                    frame = self.transforms_(frame)
                    trans_clip.append(frame)

                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)

            return clip, torch.tensor(int(class_idx))

        else:
            all_clips = []
            all_idx = []
            for i in np.linspace(self.clip_len / 2, length - self.clip_len / 2, self.test_sample_num):
                clip_start = int(i - self.clip_len / 2)
                clip = videodata[clip_start: clip_start + self.clip_len]
                if self.transforms_:
                    trans_clip = []
                    seed = random.random()
                    for frame in clip:
                        random.seed(seed)
                        frame = self.toPIL(frame)
                        frame = self.transforms_(frame)
                        trans_clip.append(frame)

                    clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                else:
                    clip = torch.tensor(clip)
                all_clips.append(clip)
                all_idx.append(torch.tensor(int(class_idx)))

            return torch.stack(all_clips), torch.tensor(int(class_idx))

