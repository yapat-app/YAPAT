import torch
import os
import pandas as pd
import numpy as np
from path import *
from species import fnjv_species
df = pd.read_csv(ANNOTATIONS_FILE)
cols = df.columns.values[3:]
spec = fnjv_species
species = np.array([s.replace('SPECIES_', '') for s in cols])
species_fnjv = np.array([s.replace('SPECIES_', '') for s in spec])

def get_test_y(validation_wav_files, nr_seconds, nr_classes = 42, species = species):
    nr_of_validation_files = len(validation_wav_files)
    labels_tensor = torch.zeros((nr_of_validation_files, nr_seconds*10, nr_classes), dtype=torch.float32)

    def time_to_index(time):
        return int(time * 10)

    for i in range(nr_of_validation_files):
        filename = validation_wav_files[i]
        path = os.path.join(STRONG_LABELS_PATH, filename + '.txt')
        with open(path, 'r') as f:
            lines = f.readlines()
        species_index = -1
        for line in lines:
            try:
                start_time, end_time, species_label = line.strip().split('\t')
                start_time_idx = time_to_index(float(start_time))
                end_time_idx = time_to_index(float(end_time))
                for index in range(nr_classes):
                    if species[index] in species_label:
                        species_index = index
                        break
            except ValueError as err:
                print('Error from file ', filename, err)
            labels_tensor[i, start_time_idx:end_time_idx + 1, species_index] = 1.0
            #temp = labels_tensor.numpy()

    return labels_tensor