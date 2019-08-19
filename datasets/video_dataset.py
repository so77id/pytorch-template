import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from tqdm import tqdm

from datasets.data_augmentation import classic_data_augmentation

from datasets.base_dataset import BaseDataset

class VideoDataset(BaseDataset):
    """Ck dataset."""

    def __init__(self, csv_file, classes, n_frames=10, type_load='last', transform=None, *args, **kwargs):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            n_frames (int): Number of frames to be loaded.
            type_load (string): 'last'  -> load the last n_frames.
                                'mid'   -> load the mid n_frames.
                                'first' -> load the first n_frames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(csv_file) is list:
            video_frames = []
            for csv in csv_file:
                if os.path.exists(csv):
                    vf = pd.read_csv(csv, sep=" ",  names=["path", "label"])
                    video_frames.append(vf)

            self.video_frame = pd.concat(video_frames, ignore_index=True)
        else:
            self.video_frame = pd.read_csv(csv_file, sep=" ", names=["path", "label"])

        self.n_frames = n_frames
        self.type_load = type_load
        self.classes = classes
        self.transform = transform

        # Load data in moment of create class
        self.clips = []
        self.labels = []
        self.clip_name = []

        print("Loading dataset")
        for i in tqdm(range(len(self.video_frame))):
            clip, label, clip_name = self.load_clip(i)
            if clip.shape[0] > 0:
                self.clips.append(clip)
                self.labels.append(label)
                self.clip_name.append(clip_name)

        # if kwargs["data_augmentation"] and kwargs["train"]:
        #     self.clips, self.labels, self.clip_name = classic_data_augmentation(self.clips, self.labels, self.clip_name)


    def __len__(self):
        return len(self.clips)

    def fix_dimensions(self, img):
        if len(img.shape) == 2:
#             img = np.expand_dims(img, axis=-1)
            new_shape = img.shape + (3,)
            new_img = np.zeros(new_shape)
            for i in range(3):
                new_img[:,:,i] = img

            return new_img
        return img

    def sub_clip(self, clip):
        if self.type_load == 'last':
            while self.n_frames > len(clip):
                clip.append(clip[-1])
            sub_clip = clip[-self.n_frames:]
        elif self.type_load == 'mid':
            while self.n_frames > len(clip):
                clip.insert(0, clip[0])
                clip.append(clip[-1])

            l_2 = len(clip)//2
            nf_2 = self.n_frames//2
            lower = max(l_2 - nf_2, 0)
            upper = min(l_2 + nf_2, len(clip))
            if self.n_frames % 2:
                if upper < len(clip):
                    upper += 1
                elif lower > 0:
                    lower -= 1

            sub_clip = clip[lower:upper]

        elif self.type_load == 'first':
            while self.n_frames > len(clip):
                clip.insert(clip[0])
            sub_clip = clip[:self.n_frames]

        return sub_clip

    def load_clip(self, idx):
        clip = []
        video_path = self.video_frame['path'][idx]
        video_label = self.video_frame['label'][idx]
        video_name = video_path.split("/")[-1]

        for parent, dirnames, filenames in os.walk(video_path):
            filenames = sorted(filenames)
            for file in filenames:
                filename = "{}/{}".format(video_path, file)
                img = io.imread(os.path.join(filename))
                img = self.fix_dimensions(img)
                clip.append(img)

        if len(clip) > 0:
            clip = self.sub_clip(clip)

        return np.array(clip), video_label, video_name

    def __getitem__(self, idx):
        clip = self.clips[idx]
        if self.transform:
            clip = self.transform(clip)

        return {'x': clip, 'y': self.labels[idx]}

    def save_videos(self, tb_writer, mode="train"):
        print("Saving dataset :{}".format(mode))
        for i in range(len(self)):
            clip = self[i]['x']
            if len(clip.shape) == 5:
                clip = clip.transpose(0,1)
                clip = clip.view(clip.size(0), -1, clip.size(3), clip.size(4))

            tb_writer.add_video("{}/{}".format(mode, self.clip_name[i]), clip.unsqueeze(0))
