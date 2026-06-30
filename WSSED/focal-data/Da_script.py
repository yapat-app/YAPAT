import pandas as pd
import numpy as np
import os
import torchaudio
import shutil
from tqdm import tqdm

path = '/Users/iliratroshani/PycharmProjects/CRNN4SED/raw_data'
ann = '/Users/iliratroshani/PycharmProjects/CRNN4SED/raw_data/weak_labels.csv'
new_path = '/Users/iliratroshani/PycharmProjects/CRNN4SED/raw_data/augmented'
new_path_labels = '/Users/iliratroshani/PycharmProjects/CRNN4SED/raw_data/augmented/strong_labels'
filepath = '/Users/iliratroshani/PycharmProjects/CRNN4SED/raw_data/augmented/weak_labels.csv'
strong_labels_path = '/Users/iliratroshani/PycharmProjects/CRNN4SED/raw_data/strong_labels'

def create_da_strong_labels():
    print('start')
    for file in os.listdir(strong_labels_path):
        if os.path.isfile(os.path.join(strong_labels_path, file)) and file.endswith('.txt'):
                shutil.copy(os.path.join(strong_labels_path, file), os.path.join(new_path_labels, '0_'+file))
                shutil.copy(os.path.join(strong_labels_path, file), os.path.join(new_path_labels, '1_' + file))
                shutil.copy(os.path.join(strong_labels_path, file), os.path.join(new_path_labels, '2_' + file))
    print('end')

def augment_data():
    df = pd.read_csv(ann)
    for file in tqdm(os.listdir(path)):
        if os.path.isfile(os.path.join(path, file)) and file.endswith('.wav'):
            row = df[df["AUDIO_FILE_ID"] == file.split('.')[0]]

            shutil.copy(os.path.join(path, file), os.path.join(new_path, '0_' + file))

            waveform, sample_rate = torchaudio.load(os.path.join(path, file), normalize=True)

            transform = torchaudio.transforms.PitchShift(sample_rate, 4)
            waveform_shift = transform(waveform)  # (channel, time)
            torchaudio.save(os.path.join(new_path, '1_' + file), waveform_shift, sample_rate)
            row["AUDIO_FILE_ID"] = '1_' + file.split('.')[0]
            df = pd.concat([df, row], ignore_index=True)

            transform2 = torchaudio.transforms.Vol(gain=1.5, gain_type="amplitude")
            waveform_vol = transform2(waveform)  # (channel, time)
            torchaudio.save(os.path.join(new_path, '2_' + file), waveform_vol, sample_rate)

            row["AUDIO_FILE_ID"] = '2_' + file.split('.')[0]
            df = pd.concat([df, row], ignore_index=True)

            # TimeStretch needs to be performed on the spectrogram.
            # Doing it directly on the waveform didn't make visible changes on the audio file and its spectrogram
            # transform3 = torchaudio.transforms.TimeStretch(fixed_rate=True)
            # w = transform3(waveform)
            # torchaudio.save(os.path.join(new_path, '3_' + file), w, sample_rate)
            # row["AUDIO_FILE_ID"] = '3_' + file.split('.')[0]
            # df = pd.concat([df, row], ignore_index=True)
    df.to_csv(filepath, index=False)


def rename_old_entries():
    df_o = pd.read_csv(ann)
    df = pd.read_csv(filepath)
    df.drop(columns=df.columns[:1], axis=1, inplace=True)
    # for file in os.listdir(path):
    #     if os.path.isfile(os.path.join(path, file)) and file.endswith('.wav'):
    #         df = df.replace([file.split('.')[0]], '0_' + file.split('.')[0])
    df.to_csv(os.path.join(new_path, 'weak_labels.csv'), index=False)

def modify_weak_labels_raw_data():
    df1 = pd.read_csv(filepath)
    selected_attributes = df1.loc[df1['subset'] == 'val', 'AUDIO_FILE_ID']

    # Extract attribute names without '0_' prefix
    attribute_names = set(attribute.replace('0_', '', 1) for attribute in selected_attributes)
    df2 = pd.read_csv(ann)
    # Insert a new column 'New_Column' with default value 'train' at index 2 in df2
    #df2.insert(2, 'subset', 'train')

    # Update rows in df2 where attribute_id matches extracted attribute names to have value 'test'
    df2.loc[df2['AUDIO_FILE_ID'].isin(attribute_names), 'subset'] = 'val'
    df2.to_csv(ann, index=False)

    print('done')

def modify_weak_lables_augmented():
    df = pd.read_csv(filepath)
    mask = (df.iloc[:, -42:].sum(axis=1) > 1) & (df['AUDIO_FILE_ID'].str.startswith('0_')) & (df['subset'] == 'test')
    valid_indices = np.where(mask)[0]
    random_indices = np.random.choice(valid_indices, size=300, replace=False)
    random_rows = df.iloc[random_indices]
    #df.insert(2, 'subset', 'train')
    df.loc[random_indices, 'subset'] = 'val'
    df.to_csv(filepath, index=False)

#df2 = pd.read_csv(ann)
#df2.drop(columns=df2.columns[:1], axis=1, inplace=True)
#df2.to_csv(ann, index=False)
#modify_weak_labels_raw_data()
#modify_weak_lables_augmented()
#rename_old_entries()
#create_da_strong_labels()
#augment_data()
print('done')

