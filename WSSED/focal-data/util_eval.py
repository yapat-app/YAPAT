import time
import species
import torch
from sklearn.metrics import classification_report, f1_score
import species
from species import target
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
from util import *
import torchaudio
from path import *
from torch import nn
sns.set()

def print_classification_report(true, pred):

    c = classification_report(true, pred, target_names=target, zero_division=0)
    print(c)
# def get_f1_score(true, pred):
#     return f1_score(true, pred, pos_label=1, average="binary")
#print('done')
def create_confusion_matrix(true, pred):
    # constant for classes
    classes = species.target
    print('creating matrix')
    # Build confusion matrix
    cf_matrix = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
def plot_all():
    df = pd.read_csv(METRICS_PATH)
    df.plot()
    plt.show()
def plot_loss(path=METRICS_PATH):
    df = pd.read_csv(path)
    plt.plot(df['Epoch'],df['train_loss'], label='Train')
    plt.plot(df['Epoch'],df['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINTS_DIR,CHECKPOINT_NAME+'_loss.png'))
    plt.show()
def plot_F1(path=METRICS_PATH):
    df = pd.read_csv(path)
    plt.plot(df['Epoch'],df['f1_1s'], label='1s')
    #plt.plot(df['Epoch'], df['f1_1s'])
    plt.plot(df['Epoch'], 100*df['global_f1'], label='Global') #same as valid_f1
    #plt.plot(df['Epoch'], 100 * df['test_1s_f1'])
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINTS_DIR, CHECKPOINT_NAME+'_F1.png'))
    plt.show()
def plot_precisionAndrecall():
    df = pd.read_csv(METRICS_PATH)
    plt.plot(df['Epoch'],df['precision_1s'])
    plt.plot(df['Epoch'], df['recall_1s'])
    plt.show()
    plt.savefig(os.path.join(CHECKPOINTS_DIR, CHECKPOINT_NAME+'_precisionAndRecall.png'))
def plot_list(first_column, list_column_name, path = METRICS_PATH):
    target.insert(0, 'epoch')
    df2 = pd.DataFrame(columns=target)
    df = pd.read_csv(path)
    for epoch in df[first_column]:
        y = df[list_column_name][epoch]
        y = y.replace('\n', '')
        y = y.replace('[', '')
        y = y.replace(']','')
        elements = y.split(' ')
        elements = list(filter(None, elements))
        elements.insert(0, epoch)
        df2.loc[epoch] = list(float(x) for x in elements)

    x_data = df2.iloc[:, 0]

    y_data = df2.iloc[:, 1:]
    num_rows = 14
    num_cols = 3

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 20))

    axes = axes.flatten()

    for i, (column, ax) in enumerate(zip(y_data.columns, axes)):
        ax.plot(x_data, y_data[column])
        ax.set_title(column)
        ax.set_xlabel(df.columns[0])
        #ax.set_ylabel(list_column_name)

    plt.tight_layout()
    plt.savefig(CHECKPOINTS_DIR+"100ep_with_da_labels.png")
    plt.show()



def plot_spectrogram(path):
    raw_transform = SpectrogramTransform(num_frames=2400, n_mels=64)

    time_mask = torchaudio.transforms.TimeMasking(
        time_mask_param=60,  # mask up to 60 consecutive time windows
    )
    freq_mask = torchaudio.transforms.FrequencyMasking(
        freq_mask_param=8,  # mask up to 8 consecutive frequency bins
    )
    train_transform = nn.Sequential(
        # Normalize(),                      # normalize so min is 0 and max is 1: Some Normaalization is done by torchaudio during load of the .wav file.
        raw_transform,
        # computes a spectrogram to have a certain number of freq bins and frames, also does AmplitudeToDb
        time_mask,  # randomly mask out a chunk of time
        freq_mask  # randomly mask out a chunk of frequencies
    )
    signal, _ = torchaudio.load(path)
    signal_mono = torch.mean(signal, dim=0, keepdim=True)
    spec_transform = torchaudio.transforms.MelSpectrogram(n_mels=64)
    signal_atod = torchaudio.transforms.AmplitudeToDB(stype='amplitude')


    #spec = raw_transform(signal_mono)
    #spec = train_transform(signal_mono)
    spec = spec_transform(signal_mono)
    spec = signal_atod(spec)
    time_axis = np.linspace(0, 60, spec.shape[-1])

    plt.figure(figsize=(15, 6))
    plt.imshow(spec[0].numpy(), cmap='plasma', origin='lower', aspect='auto',
               extent=[time_axis[0], time_axis[-1], 0, spec.shape[-2]])
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency Bin')
    plt.title('Mel Spectrogram')
    plt.savefig(CHECKPOINTS_DIR+'Spectrogram_0_INCT20955_20200314_211500.png')
    plt.show()

def plot_pvr():
    df = pd.read_csv(METRICS_PATH)
    fig, ax = plt.subplots()
    ax.plot(0.01*df['recall_1s'][:10], 0.01*df['precision_1s'][:10], color='purple')

    # add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    # display plot
    plt.show()

def get_avg(path=METRICS_PATH):
    df = pd.read_csv(path)

    print('1s F1',df['f1_1s'].mean())
    print('Global F1', df['global_f1'].mean())
    print('Precision 1s', df['precision_1s'].mean())
    print('Recall 1s', df['recall_1s'].mean())
    return df['f1_1s'].mean(), df['global_f1'].mean()

def simple_plot(y, label, path=METRICS_PATH):
    df = pd.read_csv(path)
    plt.plot(df['Epoch'], df[y], label='1 second')
    plt.plot(df['Epoch'], df['val_loss'], label='global')
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINTS_DIR, CHECKPOINT_NAME+'_1sLoss.png'))
    plt.show()

def plot_predictions(filename, path, first_column, list_column_name):
    target.insert(0, 'epoch')
    df2 = pd.DataFrame(columns=target)
    df = pd.read_csv(path)
    for epoch in df[first_column]:
        y = df[list_column_name][epoch]
        y = y.replace('\n', '')
        y = y.replace('[', '')
        y = y.replace(']', '')
        elements = y.split(' ')
        elements = list(filter(None, elements))
        elements.insert(0, epoch)
        df2.loc[epoch] = list(float(x) for x in elements)

    x_data = df2.iloc[:, 0]

    y_data = df2.iloc[:, 1:]
    num_rows = 9
    num_cols = 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 20))

    axes = axes.flatten()

    selected_columns = ['spec','BOABIS', 'BOAPRA', 'ELABIC', 'SPHSUR', 'PHYCUV', 'LEPLAT','BOAFAB','BOALEP']

    signal, _ = torchaudio.load(AUDIO_DIR+filename)
    signal_mono = torch.mean(signal, dim=0, keepdim=True)
    spec_transform = torchaudio.transforms.MelSpectrogram(n_mels=64)
    signal_atod = torchaudio.transforms.AmplitudeToDB(stype='amplitude')

    spec = spec_transform(signal_mono)
    spec = signal_atod(spec)
    time_axis = np.linspace(0, 60, spec.shape[-1])

    #plt.figure(figsize=(15, 6))
    #plt.imshow(spec[0].numpy(), cmap='plasma', origin='lower', aspect='auto',
          #     extent=[time_axis[0], time_axis[-1], 0, spec.shape[-2]])
    #plt.colorbar(format='%+2.0f dB')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Mel Frequency Bin')
    #plt.title('Mel Spectrogram')
    #plt.savefig(CHECKPOINTS_DIR + 'Spectrogram_0_INCT20955_20200314_211500.png')
    #plt.show()
    plt.subplots()

    for i, (column, ax) in enumerate(zip(selected_columns, axes)):
        if column == 'spec':
            flattened_spec = spec[0].numpy().flatten()
            ax.plot(x_data, flattened_spec, color='plasma')

            #plt.imshow(spec[0].numpy(), cmap='plasma', origin='lower', aspect='auto',
                       #extent=[time_axis[0], time_axis[-1], 0, spec.shape[-2]])
        else:
            a = x_data
            b = y_data[column]
            ax.plot(x_data, y_data[column])
            if column in ['BOAFAB', 'BOALEP']:
                ax.set_title(column, color='red')
            else:
                ax.set_title(column, color='black')
            ax.set_xlabel(df.columns[0])
            # ax.set_ylabel(list_column_name)

    #ax.imshow(spec[0].numpy(), cmap='plasma', origin='lower', aspect='auto',
             # extent=[time_axis[0], time_axis[-1], 0, spec.shape[-2]])

    plt.tight_layout()
    plt.savefig(CHECKPOINTS_DIR + "correct_and_missed_labels.png")
    plt.show()

def compare_data(file1, file2='', file3=''):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    meanf1_3s, globalf1_3s = get_avg(file1)
    meanf1_10s, globalf1_10s = get_avg(file2)
    meanf1_60s, globalf1_60s = get_avg(file3)
    f1_1s=[0, meanf1_3s, meanf1_10s, meanf1_60s]
    f1_global = [0, globalf1_3s * 100,globalf1_10s * 100,globalf1_60s*100]
    x = [0,3, 10, 60]

    # Your data
    input_sizes = [3, 10, 60]
    mean_f1_scores = [meanf1_3s, meanf1_10s, meanf1_60s]
    global_f1_scores = [globalf1_3s * 100, globalf1_10s * 100, globalf1_60s * 100]

    # Create DataFrames for Mean and Global F1 Scores
    mean_df = pd.DataFrame({'Input Size': input_sizes, 'F1 Score': mean_f1_scores, 'Metric': 'F1 Score 1 Second'})
    global_df = pd.DataFrame({'Input Size': input_sizes, 'F1 Score': global_f1_scores, 'Metric': 'Global F1 Score'})

    # Concatenate the DataFrames
    df = pd.concat([mean_df, global_df])

    # Create a grouped bar plot using Seaborn
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.set(style="whitegrid")
    ax = sns.barplot(data=df, x='Input Size', y='F1 Score', hue='Metric', palette=['green', 'orange'])
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
                    textcoords='offset points')

    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    plt.xlabel('Input Size')
    plt.ylabel('F1 Score')
    #plt.title('F1 Scores for Different Input Sizes')
    plt.legend(title='Metric', loc='lower right')


    plt.tight_layout()


    # sns.barplot(x, f1_1s, label='f1 1 seconds')
    # sns.barplot(x, f1_global, label='f1 global')
    # plt.xlabel('Input length')
    # plt.xticks(x)
    # plt.ylabel('F1 Score')
    # plt.legend()
    plt.savefig(CHECKPOINTS_DIR + "f1_input_length.png")
    plt.show()

def plot_data_statistics(filepath):
    df = pd.read_csv(filepath)
    print(df.describe())
    total = len(df.index)
    #counts = df[df.columns[3:]].sum()
    counts = df[df.columns[1:]].sum()
    counts_0 = (df[df.columns[3:]] == 0).sum()
    sorted = counts.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    present_plot= plt.bar([s.replace('SPECIES_', '') for s in sorted.index], sorted.values, label='Present', color = 'gray')
    #plt.bar(counts_0.index, counts_0.values, bottom=counts.values, label='Absent')
    plt.axhline(y = total, color='gray', linestyle='dashed', label = 'total')
    plt.xlabel('Species')
    plt.ylabel('Number of species present')
    plt.title('Distribution of class occurrences')
    plt.legend(loc = 'lower center')
    plt.xticks(rotation=90)
    for bar in present_plot:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height+20, str(int(bar.get_height())),
                              ha='center', va='bottom', fontsize=10, color='black', rotation='vertical')
    plt.tight_layout()
    plt.savefig(CHECKPOINTS_DIR + '60s_statistics.png')
    plt.show()


def plot_class_f1_scores(path = METRICS_PATH):
    df = pd.read_csv(path)
    scores = []
    for row in df.iterrows():
        val = row[1]['f1_scores_per_class']
        y = val.replace('\n', '')
        y = y.replace('[', '')
        y = y.replace(']', '')
        y = y.replace(',', '')
        elements = y.split(' ')
        f1_scores = list(float(x) for x in elements)
        f1_scores = np.array(f1_scores)
        scores.append(f1_scores)

    df2 = pd.DataFrame(scores, columns=target)
    plt.figure(figsize=(20, 10))
    sns.boxplot(data = df2.values)
    plt.title('F1 Scores Box Plot by Class')
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.xticks(range(0, 42), labels=[target[i] for i in range(42)], rotation=90)
    plt.tight_layout()
    plt.savefig(CHECKPOINTS_DIR + 'classes_f1_scores.png')
    plt.show()
    print('stop')


def get_ground_truth(filename):
    labels_tensor = torch.zeros((1, 60, 42), dtype=torch.float32)

    def time_to_index(time):
        return int(time)

    for i in range(1):
        #filename = validation_wav_files[i]
        path = os.path.join(STRONG_LABELS_PATH, filename + '.txt')
        with open(path, 'r') as f:
            lines = f.readlines()
        species_index = -1
        for line in lines:
            try:
                start_time, end_time, species_label = line.strip().split('\t')
                start_time_idx = time_to_index(float(start_time))
                end_time_idx = time_to_index(float(end_time))
                for index in range(42):
                    if target[index] in species_label:
                        species_index = index
                        break
            except ValueError as err:
                print('Error from file ', filename, err)
            labels_tensor[i, start_time_idx:end_time_idx + 1, species_index] = 1.0
    return labels_tensor


def plot_results2(filename, tens_data, x, df):
    nr_seconds = 10

    if filename == 'split2_INCT17_20191217_010000':
        # if True:
        # Load and process the audio signal
        signal, _ = torchaudio.load(AUDIO_DIR + filename + '.wav')
        signal_mono = torch.mean(signal, dim=0, keepdim=True)
        spec_transform = torchaudio.transforms.MelSpectrogram(n_mels=64)
        signal_atod = torchaudio.transforms.AmplitudeToDB(stype='amplitude')

        spec = spec_transform(signal_mono)
        spec = signal_atod(spec)
        flattened_spec = spec[0].numpy()#[:nr_seconds, :]  # Shorten spectrogram if longer than 60 seconds

        # Create a list to store non-empty subplot indices
        non_empty_subplots = []

        # Iterate through the data for each subplot and determine if it's non-empty
        for i in range(42):
            probs = []
            thresh = []
            truths = []
            plot = False

            for t in x:
                g = tens_data[t, i]
                truths.append(g)
                p = df['segment_probabilities'].iloc[t][i]
                probs.append(p)
                t = df['threshhold'].iloc[t][i]
                thresh.append(t)

                if p > t:
                    plot = True

            if plot:
                non_empty_subplots.append(i)

        # Calculate the number of rows based on the number of non-empty subplots
        num_rows = len(non_empty_subplots)

        # Create the subplots with the specified layout (two rows)
        if num_rows < 0:
            num_rows = 0
        if num_rows == 0:
            print('skipping ' + filename)
            return

        if num_rows == 1:
            print('One event detected. Skipping ' + filename)
            return

        fig, axes = plt.subplots(2, 3, figsize=(6 * num_rows, 8))
        # fig.suptitle(filename)
        ax_spec = axes[0, 0]

        time_axis = np.linspace(0, nr_seconds, flattened_spec.shape[-1])
        ax_spec.set_ylim(0, 64)

        ax_spec.imshow(flattened_spec, cmap='viridis', origin='lower', aspect='auto',
                      extent=[time_axis[0], time_axis[-1], 0, flattened_spec.shape[-2]])
        ax_spec.grid(False)

        ax_spec.set_xlabel('Time (s)', fontsize=20.0)
        ax_spec.set_ylabel('Frequency Bands (Hz)', fontsize=20.0)
        ax_spec.tick_params(axis='x', labelsize=18.0)
        ax_spec.tick_params(axis='y', labelsize=18.0)
        num_frequency_bands = flattened_spec.shape[0]

        # Set y-ticks to cover the specified range

        # Set the y-label
        #ax_spec.set_ylabel('Frequency Bands (Hz)')  # Set the y-label
        freq_max = 64  # Maximum frequency

        # Calculate the number of frequency bands in the spectrogram
        num_frequency_bands = flattened_spec.shape[0]

        # Set y-ticks to cover the entire frequency range
        #y_ticks = np.arange(0, num_frequency_bands, 10)

        #ax_spec.set_yticks(y_ticks)
        # Iterate through non-empty subplots and populate each subplot
        for idx, i in enumerate(non_empty_subplots):
            col = idx  # Column index for the current subplot

            # Plot the mel spectrogram in the first row

            # ax_spec.set_title('Mel Spectrogram')
            if col == 0:
                ax_lines = axes[0, 1]
            if col == 1:
                ax_lines = axes[0, 2]
            if col == 2:
                ax_lines = axes[1, 0]
            if col == 3:
                ax_lines = axes[1, 1]
            if col == 4:
                ax_lines = axes[1, 2]
            # Plot the three lines (probs, thresh, truths) in the second row
            #ax_lines = axes[1, col]

            probs = []
            thresh = []
            truths = []

            for t in x:
                g = tens_data[t, i]
                truths.append(g)
                p = df['segment_probabilities'].iloc[t][i]
                probs.append(p)
                t = df['threshhold'].iloc[t][i]
                thresh.append(t)

            ax_lines.set_facecolor('white')
            #ax_lines.grid(color='gray', linestyle='--')
            ax_lines.plot(x, probs, color='green', label='Probability')
            ax_lines.plot(x, thresh, linestyle='--', color='orange', label='Threshold')

            ax_lines.bar(x, truths, color='lightgray', label='Truth', alpha=0.7)
            ax_lines.set_title(target[i], size=20.0)
            ax_lines.legend(loc='lower right', fontsize=18.0)
            ax_lines.set_xlim(0, nr_seconds)
            ax_lines.set_xlabel('Time (s)', fontsize=20.0)
            ax_lines.set_ylabel('Probability', fontsize=20.0)
            ax_lines.tick_params(axis='x', labelsize=18.0)
            ax_lines.tick_params(axis='y', labelsize=18.0)

        # Adjust the layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(os.path.join(CHECKPOINTS_DIR, filename + 'xxx_results.png'))
        plt.show()


def plot_results(filename, tens_data, x, df):
    #nr_seconds = 60
    nr_seconds = 10

    if filename == 'split2_INCT17_20191217_010000':
    #if True:
        # Load and process the audio signal
        signal, _ = torchaudio.load(AUDIO_DIR + filename + '.wav')
        signal_mono = torch.mean(signal, dim=0, keepdim=True)
        spec_transform = torchaudio.transforms.MelSpectrogram(n_mels=64)
        signal_atod = torchaudio.transforms.AmplitudeToDB(stype='amplitude')

        spec = spec_transform(signal_mono)
        spec = signal_atod(spec)
        flattened_spec = spec[0].numpy()[:nr_seconds, :]  # Shorten spectrogram if longer than 60 seconds

        # Create a list to store non-empty subplot indices
        non_empty_subplots = []

        # Iterate through the data for each subplot and determine if it's non-empty
        for i in range(42):
            probs = []
            thresh = []
            truths = []
            plot = False

            for t in x:
                g = tens_data[t, i]
                truths.append(g)
                p = df['segment_probabilities'].iloc[t][i]
                probs.append(p)
                t = df['threshhold'].iloc[t][i]
                thresh.append(t)

                if p > t:
                    plot = True

            if plot:
                non_empty_subplots.append(i)

        # Calculate the number of rows based on the number of non-empty subplots
        num_rows = len(non_empty_subplots)

        # Create the subplots with the specified layout (two rows)
        if num_rows < 0:
            num_rows = 0
        if num_rows == 0:
            print('skipping '+ filename)
            return

        if num_rows == 1:
            print('One event detected. Skipping ' + filename)
            return

        fig, axes = plt.subplots(2, num_rows, figsize=(6 * num_rows, 8))
        #fig.suptitle(filename)

        # Iterate through non-empty subplots and populate each subplot
        for idx, i in enumerate(non_empty_subplots):
            col = idx  # Column index for the current subplot

            # Plot the mel spectrogram in the first row
            ax_spec = axes[0, col]
            time_axis = np.linspace(0, nr_seconds, flattened_spec.shape[-1])

            ax_spec.imshow(flattened_spec, cmap='viridis', origin='lower', aspect='auto',
                           extent=[time_axis[0], time_axis[-1], 0, flattened_spec.shape[-2]])

            #ax_spec.set_title('Mel Spectrogram')

            # Plot the three lines (probs, thresh, truths) in the second row
            ax_lines = axes[1, col]

            probs = []
            thresh = []
            truths = []

            for t in x:
                g = tens_data[t, i]
                truths.append(g)
                p = df['segment_probabilities'].iloc[t][i]
                probs.append(p)
                t = df['threshhold'].iloc[t][i]
                thresh.append(t)

            ax_lines.set_facecolor('white')
            ax_lines.grid(color='gray', linestyle='--')
            ax_lines.plot(x, probs, color='green', label='Probability')
            ax_lines.plot(x, thresh,linestyle='--', color='orange', label='Threshold')

            ax_lines.bar(x, truths, color='lightgray', label='Truth', alpha=0.7)
            ax_lines.set_title(target[i])
            ax_lines.legend(loc='lower right')
            ax_lines.set_xlim(0, nr_seconds)

        # Adjust the layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(os.path.join(CHECKPOINTS_DIR, filename + '_results.png'))
        plt.show()


def plot_result(filename):
    path = os.path.join(CHECKPOINTS_DIR,'11669_100ep_10sinput_without_da_with_maskingnet_light_model_test_results.csv')
    df = pd.read_csv(path)
    groups = df.groupby('filename')

    for group in groups:
        file = group[0]
        file_ground_truth = get_ground_truth(file)[0]
        ground_truth_data = file_ground_truth.numpy()
        df = group[1]
        x = df['time'].values

        df['segment_probabilities'] = df['segment_probabilities'].apply(
            lambda x: [float(val) for val in x.strip('[]').split()])
        df['threshhold'] = df['threshhold'].apply(lambda x: [float(val) for val in x.strip('[]').split()])
        plot_results2(file, ground_truth_data, x, df)
        #continue
        #inspect_test_results(x, df, ground_truth_data, file)


    # for idx, row in df.iterrows():
    #     plt.plot(row['time'], row['segment_probabilities'][idx], label=f'Sample {idx + 1}', alpha=0.5)
    #
    # plt.xlabel('Time')
    # plt.ylabel('Values')
    # plt.legend(loc='upper right')
    #
    # # Create subplots for 'threshold'
    # plt.subplot(2, 1, 2)  # 2 rows, 1 column, subplot 2
    # plt.title('Thresholds Over Time')
    #
    # for idx, row in df.iterrows():
    #     plt.plot(row['time'], row['threshold'], linestyle='--', label=f'Threshold {idx + 1}', alpha=0.5)
    #
    # plt.xlabel('Time')
    # plt.ylabel('Values')
    # plt.legend(loc='upper right')
    #
    # # Adjust layout
    # plt.tight_layout()
    #
    # # Show the plot
    # plt.show()


def inspect_test_results(x, df, ground_truth_data, filename2):
    for i in range(42):
        probs = []
        thresh = []
        truths = []
        plot = False
        for t in x:
            g = ground_truth_data[t, i]
            truths.append(g)
            p = df['segment_probabilities'].iloc[t][i]
            probs.append(p)
            t = df['threshhold'].iloc[t][i]
            thresh.append(t)
            if p > t:
                plot = True
        plt.subplot(2, 1, 1)
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(filename2)
        if plot:
            ax1.plot(x, probs, color='blue', label=target[i])
            ax1.plot(x, thresh, color='orange')  # , label=target[i])
            ax1.legend(loc='upper right')

            # plt.title(filename2)
        ax2.plot(x, truths, color='red', label=target[i])
        ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def view_test_results():

                                         #11669_100ep_60sinput_without_da_with_maskingnet_light_model_test_results.csv
    path = os.path.join(CHECKPOINTS_DIR, '11669_100ep_60sinput_without_da_with_maskingnet_light_model_test_results_macro.csv')
    df = pd.read_csv(path)
    groups = df.groupby('filename')

    for group in groups:
        file = group[0]
        file_ground_truth = get_ground_truth(file)[0]
        ground_truth_data = file_ground_truth.numpy()
        df = group[1]
        x = df['time'].values

        df['segment_probabilities'] = df['segment_probabilities'].apply(
            lambda x: [float(val) for val in x.strip('[]').split()])
        df['threshhold'] = df['threshhold'].apply(lambda x: [float(val) for val in x.strip('[]').split()])
        plot_results(file, ground_truth_data, x, df)
        #inspect_test_results(x, df, ground_truth_data, file)
        print(file)

#plot_list('Epoch','threshold')#, path='/Users/iliratroshani/PycharmProjects/CRNN4SED/checkpoints/75e_without_da_metrics.csv')
#simple_plot('total_loss_batch_1s', 'Validation loss')
#plot_list('time','segment_probabilities', TEST_RESULTS_PATH)
plot_loss('/Users/iliratroshani/PycharmProjects/CRNN4SED/checkpoints/11788_100ep_FNJV_var_sinput_without_da_with_masking_net_light_model_metrics.csv')

#plot_F1()
#plot_spectrogram(os.path.join('/Users/iliratroshani/PycharmProjects/CRNN4SED/raw_data/augmented', '0_INCT20955_20200314_211500.wav'))
#plot_predictions(filename='INCT20955_20191012_041500.wav',first_column='time',list_column_name='segment_probabilities', path=TEST_RESULTS_PATH)
#plot_result('split2_INCT17_20191217_010000')
#view_test_results()
#get_avg(path=CHECKPOINTS_DIR+'11569_100ep_with_da_net_light_model_metrics.csv')
#get_avg(path=CHECKPOINTS_DIR+'11569_100ep_with_da_net_light_model_metrics_max.csv')
#get_avg(path=CHECKPOINTS_DIR+'11569_100ep_with_da_net_light_model_metrics_exp.csv')
#plot_loss(path=CHECKPOINTS_DIR+'11569_100ep_without_da_without_masking_net_light_model_metrics.csv')
#plot_F1(path=CHECKPOINTS_DIR+'11569_100ep_without_da_without_masking_net_light_model_metrics.csv')
#simple_plot('total_loss_batch_1s', 'Validation loss',path=CHECKPOINTS_DIR+'11569_100ep_without_da_without_masking_net_light_model_metrics.csv')
#plot_list('time','species', TEST_RESULTS_PATH)
#plot_precisionAndrecall()

#get_avg()
#file3s = os.path.join(CHECKPOINTS_DIR,'11669_100ep_3sinput_without_da_with_maskingnet_light_model_metrics_val.csv')
#file10s = os.path.join(CHECKPOINTS_DIR,'11669_100ep_10sinput_without_da_with_maskingnet_light_model_metrics_val.csv')
#file60s = os.path.join(CHECKPOINTS_DIR,'11669_100ep_60sinput_without_da_with_maskingnet_light_model_metrics_val.csv')
#file60s = os.path.join(CHECKPOINTS_DIR, '11569_100ep_without_da_without_masking_net_light_model_metrics.csv')
#get_avg(file10s)
#get_avg(file60s)
#compare_data(file3s, file10s, file60s)
#plot_data_statistics(os.path.join(BASE_PATH + 'raw_data/10_splits/weak_labels.csv'))
#plot_data_statistics(os.path.join(BASE_PATH+ 'raw_data/weak_labels.csv'))
#plot_class_f1_scores('/Users/iliratroshani/PycharmProjects/CRNN4SED/checkpoints/final3/11669_100ep_60sinput_without_da_with_maskingnet_light_model_metrics.csv')
#plot_class_f1_scores()
#plot_data_statistics(os.path.join(BASE_PATH+ 'raw_data/FNJV/weak_labels.csv'))