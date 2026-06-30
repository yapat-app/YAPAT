import os
import torch
import torchaudio
import pandas as pd
import path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from Audio_Dataset import add_noise_and_extend

class AnuraSet(Dataset):
    """ AnuraSet: A dataset for bioacoustic classification of tropical anurans

    Args:
        annotations_file (string): path of the metadata csv table with labels of
            AnuraSet and audio samples information
        audio_dir(string): path of the folder with audio samples of the AnuraSet
            associated with the metadata table
        transformation (callable?): L A function/transform that takes audios before
            feature extraction and returns a transformed version.This tranformantions
            include melspectogram and augmentations.
        device (string): if using cuda (GPU) or CPU
        train (bool): If True, creates dataset from using 'train' samples in the
                'subset' column of the metadata
    """

    def __init__(self,
                 data,
                 annotations_file,
                 audio_dir,
                 transformation,

                 train=True,
                 val = False,
                 idx = 0,
                 ):
        if data is None:
            if isinstance(annotations_file, str):
                df = pd.read_csv(annotations_file)
            else:
                df = annotations_file.copy()

            if "subset" in df:
                if train:
                    df = df[df["subset"] == "train"]

                elif val:
                    df = df[df["subset"] == "val"]
                else:
                    df = df[df["subset"] == "test"]
        else:
            df = data

        self.annotations = df
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.idx = idx
        self.filename = df["AUDIO_FILE_ID"].to_numpy()

    def __len__(self):
        return len(self.annotations)

        #return len(self.annotations) + len(self.annotations) * self.num_augmented_samples

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, _ = torchaudio.load(audio_sample_path)
        # Handle stereo files
        signal_mono = torch.mean(signal, dim=0, keepdim=True)
        signal_mono = self.transformation(signal_mono)
        # Create a 3-channel input
        if path.MODEL_NAME != 'net_light_model':
            signal_mono = signal_mono.repeat(3, 1, 1)

        return signal_mono, label, index

    def _get_audio_sample_path(self, index):
        ext=''
        if self.idx != 0:
            ext = '.wav'
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, self.idx]+ext)
        return path

    def _get_audio_sample_label(self, index):
        labels = self.annotations.iloc[index, 3:]
        # replace all species occurrencies that are greater than 1 with one since we want a
        # binary classification problem with weak labels
        one_hot_encoded_row = (labels >= 1).astype(int)
        #labels.mask[labels > 1, 1]
        #return torch.Tensor(self.annotations.iloc[index, 2:])
        return torch.Tensor(one_hot_encoded_row)
