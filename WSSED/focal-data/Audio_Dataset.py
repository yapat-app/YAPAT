import math
import os
import scrap_script
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset
from util import SpectrogramTransform, add_noise_and_extend, AudioAugmentationTransform
import path
from torchvision.transforms import Resize
class Audio_Dataset(Dataset):
    def __init__(self, audio_folder, labels_df, transform=None, fixed_length = None):
        self.audio_folder = audio_folder
        self.labels_df = labels_df
        self.transform = transform
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        filename = self.labels_df.iloc[idx]['Arquivo do registro']
        filepath = os.path.join(self.audio_folder, filename)
        signal, sample_rate = torchaudio.load(filepath)
        waveform = torch.mean(signal, dim=0, keepdim=True)

        if self.fixed_length != None:
            extended_waveform = add_noise_and_extend(filepath, self.fixed_length)
            input_length_in_seconds_f = self.fixed_length
            waveform = extended_waveform
        else:
            # Calculate the length in seconds
            input_length_in_seconds_f = float(waveform.size(1)) / float(sample_rate)

        input_length_in_seconds = round(input_length_in_seconds_f)

        # Apply transform if available
        # if self.transform:
        #     waveform = self.transform(waveform)
        # else:
        if True:
            raw_transform = SpectrogramTransform(num_frames=input_length_in_seconds * 40, n_mels=64)
            resamp = torchaudio.transforms.Resample(new_freq=22050)
            time_mask = torchaudio.transforms.TimeMasking(
                time_mask_param=60,  # mask up to 60 consecutive time windows
            )
            freq_mask = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=8,  # mask up to 8 consecutive frequency bins
            )
            train_transform = nn.Sequential(
                resamp,
                raw_transform,
                #time_mask,
                #freq_mask
            )#, Resize(size =(128, 1024)))#(250,450)))#250,900 was worse
            self.transform = train_transform
            waveform = self.transform(waveform)
            if path.MODEL_NAME != 'net_light_model':
                waveform = waveform.repeat(3, 1, 1)


        # Retrieve label from the dataframe
        label_df = self.labels_df.iloc[idx, 1:]
        label = torch.Tensor(label_df)
        return waveform, label

