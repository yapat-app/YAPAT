from sqlite3 import DatabaseError
from tkinter.dnd import DndHandler
import xdrlib
from pytest import xfail
from torch.utils.data import Dataset
from MainClasses.DataHandler import DataHandler
from h5py import File
import pandas as pd
from MainClasses.transforms import *
import os
import random

def parse_transform(transform):
    transforms = []
    for trans in transform:
        if trans == 'shift':
            transforms.append(TimeShift(0, 20))
        elif trans == 'freqmask':
            transforms.append(FreqMask(2, 6))
        elif trans == 'timemask':
            transforms.append(TimeMask(1, 20))
    return torch.nn.Sequential(*transforms)


class DCASE2017(Dataset):
    def __init__(self, options, type_, transform=None):
        super(DCASE2017, self).__init__()
        self.dh = DataHandler(dataset=options.dataset)
        self.K = options.num_events
        self.x_data, self.y_data, self.audio_files = self.dh.load_dcase17(type_)
        self.type_ = type_

    def __getitem__(self, item):
        x_data = self.x_data[item]
        y_data = self.y_data[item]
        audio_file = self.audio_files[item]
        if self.type_ == 'training':
            y_tagging = y_data
            return x_data, y_tagging, audio_file
        else:
            return x_data, y_data, audio_file

    def __len__(self):
        return len(self.x_data)


class DCASE2018(Dataset):
    def __init__(self, type_, transform=None):
        super(DCASE2018, self).__init__()
        self.dh = DataHandler(dataset='DCASE2018')
        self.x_data, self.y_data = self.dh.load_dcase18(type=type_)

    def __len__(self):
        return len(self.x_data['filenames'])

    def __getitem__(self, item):
        x_file = self.x_data['filenames'][item]
        x_data = self.x_data['data'][item].transpose()
        y_data = self.y_data['labels'][self.y_data['filenames'].index(x_file.split('/')[-1])]
        return x_data, y_data, x_file


class DCASE2019(Dataset):
    def __init__(self, type_, transform=None):
        super(DCASE2019, self).__init__()
        self.dh = DataHandler(dataset='DCASE2019')
        self.x_data, self.y_data = self.dh.load_dcase19(type=type_)
        self.transform = transform
    def __len__(self):
        return len(self.x_data['filenames'])

    def __getitem__(self, item):
        x_file = self.x_data['filenames'][item]
        x_data = self.x_data['data'][item].transpose()
        y_data = self.y_data['labels'][self.y_data['filenames'].index(x_file.split('/')[-1])]
        if not self.transform is None:
            trans = parse_transform(self.transform)
            x_data = trans(torch.from_numpy(x_data))
        return x_data, y_data, x_file


class AnuraSet(Dataset):
    def __init__(self, options, type_, transform=None):
        super(AnuraSet, self).__init__()
        self.dh = DataHandler(dataset="AnuraSet")
        self.x_data, self.y_data, self.audio_files = self.dh.load_anuraset(type_=type_)
        self.type_ = type_

    def __getitem__(self, item):
        x_data = self.x_data[item]
        y_data = self.y_data[item]
        audio_file = self.audio_files[item]
        return x_data, y_data, audio_file

    def __len__(self):
        return len(self.x_data)
    
    
