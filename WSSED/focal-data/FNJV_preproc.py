import scrap_script
import scrap_script as sc
import species as species
from path import *
import pandas as pd
from util import SpectrogramTransform
import torchaudio
import torchaudio.transforms as T
from torch import nn
print(AUDIO_DIR)

def get_labels_FNJV_filtered():
    metadata = pd.read_csv(AUDIO_DIR + 'FNJV/metadata.csv', delimiter=";")
    label_set = metadata.copy()
    label_set = label_set.loc[label_set["Classe"] == "Amphibia", ["Arquivo do registro", "Gênero", "Espécie"]]
    label_set.loc[label_set["Espécie"] == "cf. podicipinus", "Espécie"] = "podicipinus"
    label_set = label_set[label_set["Arquivo do registro"].str.endswith('.wav')]

    label_set['species'] = label_set['Gênero'].astype(str).str[:3] + label_set['Espécie'].astype(str).str[:3]
    label_set['species'] = 'SPECIES_' + label_set['species'].str.upper()
    filteres = label_set.drop_duplicates(["Gênero", "Espécie"])
    new = []
    for s in species.target:
        new.append('SPECIES_' + s)


    # Create dummy columns for each species
    dummies = pd.get_dummies(label_set['species']).astype(int)

    # Concatenate 'name' column from original DataFrame with dummy columns
    new_df = pd.concat([label_set['Arquivo do registro'], dummies], axis=1)
    new_df.to_csv(AUDIO_DIR + 'FNJV/weak_labels.csv', index=False)


def get_labels_FNJV():
    metadata = pd.read_csv(AUDIO_DIR+'FNJV/metadata.csv', delimiter=";")
    label_set = metadata.copy()
    label_set = label_set.loc[label_set["Classe"] == "Amphibia", ["Arquivo do registro","Gênero", "Espécie"]]
    label_set.loc[label_set["Espécie"] == "cf. podicipinus", "Espécie"] = "podicipinus"
    label_set = label_set[label_set["Arquivo do registro"].str.endswith('.wav')]

    label_set['species'] = label_set['Gênero'].astype(str).str[:3] + label_set['Espécie'].astype(str).str[:3]
    label_set['species'] = 'SPECIES_'+label_set['species'].str.upper()
    filteres = label_set.drop_duplicates(["Gênero", "Espécie"])
    new = []
    for s in species.target:
        new.append('SPECIES_' + s)

    df_new= pd.DataFrame(0, index = label_set.index, columns=['name']+new)

    for index, row in label_set.iterrows():
        if row['Arquivo do registro'].endswith('.wav'):
            spec = row['species']
            df_new.loc[index,'name'] = row['Arquivo do registro']
            df_new.loc[index, spec] = 1
    df_new.drop(df_new[df_new['name'] == 0].index)

    df = pd.read_csv(ANNOTATIONS_FILE)
    spec = species.fnjv_species#filteres['species']
    cond = df[spec].any(axis=1)
    filtered_a = df[cond]
    #spec_col = filtered_a.columns[3:]
    #clipped = filtered_a[spec_col].clip(0, 1)

    df_new.to_csv(AUDIO_DIR + 'FNJV/weak_labels.csv', index=False)



    print('stop')


import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import random as RANDOM

def add_noise_and_extend(audio_file, target_duration=900, noise_level=0.1):
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Calculate the current duration of the audio file
    current_duration = float(waveform.size(1)) / float(sample_rate)
    input_length_in_seconds_f = float(waveform.size(1)) / float(sample_rate)

    # Calculate the duration difference to reach the target duration
    duration_diff = target_duration - current_duration
    sig = waveform[0]
    if current_duration < target_duration:
        split = np.hstack((sig, scrap_script.noise(sig, (int(sample_rate * target_duration) - len(sig)), 0)))
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
    output_file = 'extended_audio_noise.wav'
    sf.write(output_file, waveform_extended.numpy(), int(sample_rate))

    return output_file

def repeat_and_extend(audio_file, target_duration=900, noise_level=0.1):
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Calculate the current duration of the audio file
    current_duration = float(waveform.size(1)) / float(sample_rate)
    input_length_in_seconds_f = float(waveform.size(1)) / float(sample_rate)

    # Calculate the duration difference to reach the target duration
    duration_diff = target_duration - current_duration
    sig = waveform[0]
    # if current_duration < target_duration:
    #     split = np.hstack((sig, scrap_script.noise(sig, (int(sample_rate * target_duration) - len(sig)), 0.5)))
    #     split = torch.tensor(split)
    #
    # # Add the noise to the audio
    # waveform_with_noise = split #torch.cat([waveform.squeeze(), noise])
    waveform_with_noise = sig

    # Extend or trim the audio to reach the target duration
    if duration_diff > 0:
        # Extend the audio by repeating it to reach the target duration
        repetitions = int(np.ceil(duration_diff / current_duration))
        waveform_extended = waveform_with_noise.repeat(repetitions)
        # Trim to the target duration
        waveform_extended = waveform_extended[:int(sample_rate * target_duration)] # the resulting file length does not correspond to the target file length in this calculation
    else:
        # Trim the audio to the target duration
        waveform_extended = waveform_with_noise[:int(sample_rate * target_duration)]

    # Save the modified audio to a new file
    output_file = 'extended_audio_repeat.wav'
    sf.write(output_file, waveform_extended.numpy(), int(sample_rate))

    return output_file


# Usage example:
#input_audio_file = '/Users/iliratroshani/PycharmProjects/CRNN4SED/raw_data/FNJV/FNJV_0032312_Boana_faber_Sao Luis do Paraitinga_SP_Lucas Rodriguez Forti.wav'
input_audio_file = '/Users/iliratroshani/PycharmProjects/CRNN4SED/raw_data/FNJV/FNJV_0013104_Dendropsophus_minutus_Campinas_SP_Celio Fernando Baptista Haddad.wav'
target_duration = 900  # 10 minutes in seconds
noise_level = 0.1  # Adjust the noise level as needed

output_file1 = add_noise_and_extend(input_audio_file, target_duration, noise_level)
output_file2 = repeat_and_extend(input_audio_file, target_duration, noise_level)
print(f"Audio extended to 10 minutes with noise: {input_audio_file}")

print('start')
#sc.split_audio_files(60, 1)
#get_labels_FNJV()
#get_labels_FNJV_filtered()
def filter_anuranset():
    df = pd.read_csv(ANNOTATIONS_FILE)
    spec = species.fnjv_species  # filteres['species']
    a = df[df.columns[:3]]
    b = df[df.columns[df.columns.isin(spec)]]
    filtered_df = pd.concat([a, b], axis=1)
    cond = filtered_df[spec].any(axis=1)
    filtered_a = filtered_df[cond]
    print('stop')


#filter_anuranset()
print('done')

