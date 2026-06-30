import math
import numpy as np
import scrap_script
import torch
import torchaudio
import torchaudio.transforms as T
import os
import pandas as pd
from tqdm import tqdm
import warnings as w
import random as RANDOM

w.filterwarnings("ignore")

class SpectrogramTransform(torch.nn.Module):
    def __init__(self, num_frames=400, n_mels=64):
        super(SpectrogramTransform, self).__init__()
        self.num_frames = num_frames
        self.n_mels = n_mels

    def forward(self, waveform):
        hop_length_f = waveform.size(1) // self.num_frames
        hop_length = round(hop_length_f)
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=waveform.size(-1),
            n_fft=hop_length * 2,
            hop_length=hop_length,
            n_mels=self.n_mels
        )(waveform)

        spectrogram_db = T.AmplitudeToDB(stype='amplitude')(mel_spectrogram)

        # Set exact number of frames because there is +/-2 and the dataloader complains during iteration
        desired_frames = self.num_frames
        current_frames = spectrogram_db.size(2)

        if current_frames < desired_frames:
            spectrogram_db = torch.nn.functional.pad(spectrogram_db, (0, desired_frames - current_frames))
            print('padded with 0')
        elif current_frames > desired_frames:
            spectrogram_db = spectrogram_db[..., :desired_frames]
        spectrogram_db.to('cpu')
        return spectrogram_db


class AudioAugmentationTransform(torch.nn.Module):
    def __init__(self, target_duration=600):
        super(AudioAugmentationTransform, self).__init__()
        self.target_duration = target_duration

    def forward(self, waveform):
        # Apply the augmentation to the input waveform
        extended_audio = self.add_noise_and_extend(waveform)
        return extended_audio

    def add_noise_and_extend(self, waveform):
        sample_rate = waveform.shape[-1]

        # Calculate the current duration of the audio file
        current_duration = waveform.size(1) / sample_rate

        # Calculate the duration difference to reach the target duration
        duration_diff = self.target_duration - current_duration
        sig = waveform[0]
        if current_duration < self.target_duration:
            split = np.hstack((sig, scrap_script.noise(sig, (int(sample_rate * self.target_duration) - len(sig)), 0.5)))
            split = torch.tensor(split)

        # Add the noise to the audio
        waveform_with_noise = split #torch.cat([waveform.squeeze(), noise])

        # Extend or trim the audio to reach the target duration
        if duration_diff > 0:
            # Extend the audio by repeating it to reach the target duration
            repetitions = int(np.ceil(duration_diff / current_duration))
            waveform_extended = waveform_with_noise.repeat(repetitions)
            # Trim to the target duration
            waveform_extended = waveform_extended[:int(sample_rate * self.target_duration)]
        else:
            # Trim the audio to the target duration
            waveform_extended = waveform_with_noise[:int(sample_rate * self.target_duration)]

        return waveform_extended.unsqueeze(0)

# Code from Birdnet Repository
def noise(sig, shape, amount=None):
    """Creates noise.

    Creates a noise vector with the given shape.

    Args:
        sig: The original audio signal.
        shape: Shape of the noise.
        amount: The noise intensity.

    Returns:
        An numpy array of noise with the given shape.
    """
    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.5)

    # Create Gaussian noise
    try:
        noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)
    except:
        noise = np.zeros(shape)

    return noise.astype("float32")

def get_strong_labels():
    file_list = os.listdir('./raw_data/strong_labels')
    data = pd.DataFrame(columns=["AUDIO_FILE_ID", "start", "end", "species"])
    for file in tqdm(file_list):
        f = os.path.splitext(file)[0]
        file_path = os.path.join('./raw_data/strong_labels', file)
        try:
            with open(file_path, 'r') as file:
                # ToDo this is inefficient
                for line in file:
                    columns = line.strip().split('\t')
                    #data['AUDIO_FILE_ID'].append(f)
                    #data['start'].append(columns[0])
                    data.loc[len(data)] = {"AUDIO_FILE_ID":f, "start":columns[0], "end":columns[1], "species":columns[2]}
        except Exception as ex:
            data.loc[len(data)] = {"AUDIO_FILE_ID": f, "start": 0, "end": 0, "species": "NaN"}
            print("failed to add ", f, ex)
    return data

def add_noise_and_extend(audio_file, target_duration, output_file = None, new_sample_rate = None):
    waveform, sample_rate = torchaudio.load(audio_file)
    if new_sample_rate is not None:
        sample_rate = new_sample_rate
        resampler = torchaudio.transforms.Resample(new_freq=new_sample_rate)
        waveform = resampler(waveform)
    waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Calculate the current duration of the audio file
    current_duration = float(waveform.size(1)) / float(sample_rate)

    # Calculate the duration difference to reach the target duration
    duration_diff = target_duration - current_duration
    sig = waveform[0]
    if current_duration > target_duration:
        print(audio_file)
        waveform_with_noise = sig
    if current_duration <= target_duration:
        split = np.hstack((sig, noise(sig, (int(sample_rate * target_duration) - len(sig)), 0)))
        split = torch.tensor(split)

        # Add the noise to the audio
        waveform_with_noise = split #torch.cat([waveform.squeeze(), noise])

    # Extend or trim the audio to reach the target duration
    if duration_diff > 0:

        # Extend the audio by repeating it to reach the target duration
        repetitions = int(np.ceil(duration_diff / current_duration))
        waveform_extended = waveform_with_noise.repeat(repetitions)
        # Trim to the target duration
        waveform_extended = waveform_extended[:int(sample_rate * target_duration)]
    else:
        # Trim the audio to the target duration
        waveform_extended = waveform_with_noise[:int(sample_rate * target_duration)]

    # Save the modified audio to a new file
    if output_file is not None:
        import soundfile as sf
        sf.write(output_file, waveform_extended.numpy(), int(sample_rate))

    return waveform_extended.unsqueeze(0)