from cgitb import strong
import librosa as lrs
import numpy as np
import pickle as pkl
from scipy import signal
import os
from tqdm import tqdm
import random
from h5py import File
import pandas as pd
import json
import config as mc
'''
This class defines the operations of datasets.
'''

class DataHandler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.config = json.load(open('./MainClasses/config.json', 'r'))[self.dataset]
        if self.dataset == "AnuraSet":
            self.config["data"]["sr"] = mc.sample_rate
            self.config["data"]["sample_width"] = mc.n_fft
            self.config["data"]["sample_step"] = mc.hop_length
            self.config["data"]["n_mels"] = mc.n_mels
            if mc.BAG_SECONDS == "full":
                self.config["data"]["clip_len"] = 60
            else:
                self.config["data"]["clip_len"] = mc.BAG_SECONDS
            self.config["data"]["labels"] = mc.TARGET_SPECIES or self.config["data"]["labels"]
            self.config["eval"].update(mc.ANURASET_EVAL)
        self.root = self.config['data']['root']
        self.sr = self.config['data']['sr']
        self.frame_width = self.config['data']['sample_width']
        self.frame_step = self.config['data']['sample_step']
        self.n_mels = self.config['data']['n_mels']
        self.clip_len = self.config['data']['clip_len']

    # pre-emphasize before MFCCs extraction
    def __emphasize__(self, s):
        emphasized_s = np.append(s[0], s[1:] - 0.97 * s[:-1])
        return emphasized_s

    # enframe a signal to several frames.
    def __enframe__(self, s):
        sample_width = int(self.frame_width * self.sr)
        step = int(self.frame_step * self.sr)
        slength = len(s)
        nframes = int(np.ceil((1.0 * slength - sample_width + step) / step) + 1)
        pad_length = int((nframes - 1) * step + sample_width)
        zeros = np.zeros((pad_length - slength,))
        pad_signal = np.concatenate((s, zeros))
        indices = np.tile(np.arange(0, sample_width), (nframes, 1)) + \
                  np.tile(np.arange(0, nframes * step, step), (sample_width, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = pad_signal[indices]
        return frames, nframes

    # add window function to every single frame
    def __windowing__(self, frames):
        frames_win = frames * signal.windows.hamming(int(self.frame_width * self.sr))
        return frames_win

    # normalize the data to [0,1]
    def normalize(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        return data

    def get_absolute_path(self, parent_path, filter_by=None, get_by=None, numbers=0, shuffle=False):
        if not shuffle:
            files = sorted(os.listdir(parent_path))
        else:
            files = os.listdir(parent_path)
            random.shuffle(files)
        absolute_path = []
        for f in files:
            if get_by is None and filter_by is None:
                absolute_path.append(os.path.join(parent_path, f))
            elif filter_by is not None and get_by is None:
                if filter_by in f:
                    continue
                else:
                    absolute_path.append(os.path.join(parent_path, f))
            elif get_by is not None and filter_by is None:
                if get_by in f:
                    absolute_path.append(os.path.join(parent_path, f))
                else:
                    continue
            else:
                return 0
        if numbers == 0:
            return absolute_path
        else:
            return random.sample(absolute_path, numbers)

    def standardization(self, x):
        return (x - x.mean()) / np.std(x)

    def load_dcase17(self, type):
        files = self.get_absolute_path('/data0/gaolj/sed_data/DCASE2017/features_base/' + type)
        with open('/data0/gaolj/sed_data/DCASE2017/features_base/' + type + '_label.pkl', 'rb') as f:
            file = pkl.load(f)
            labels = file['loc_labels']
            audio_files = file['audio_files']

        data = []
        files_bar = tqdm(files)
        n = 0
        del_n = []
        nframe = int(np.floor((self.clip_len * self.sr - self.frame_width) / self.frame_step) + 1)
        for f in files_bar:
            files_bar.set_description('Loading {} dataset {}/{}'.format(type, n, len(files)))
            data_ = pkl.load(open(f, 'rb'))
            data_ = data_.T
            if np.std(data_) == 0:
                del_n.append(n)
                n += 1
                continue
            if len(data_) < nframe:
                res_n = (nframe - len(data_))
                if res_n > len(data_):
                    del_n.append(n)
                    n += 1
                    continue
                res = data_[len(data_) - res_n:]
                data_ = np.concatenate([data_, res])
                if type != 'training':
                    actual_label = labels[n, :len(data_) - res_n, :]
                    res = actual_label[len(actual_label) - (nframe - len(actual_label)):, :]
                    labels[n, len(actual_label):len(actual_label) + res_n, :] = res
            data.append(self.standardization(data_))
            n += 1
        labels = np.delete(labels, del_n, 0)
        audio_files = np.delete(audio_files, del_n, 0)
        return data, labels, audio_files

    def load_urbansed(self, type):
        data_file = os.path.join(self.root, 'features', type + '.pkl')
        if type in ['train']:
            label_file = os.path.join(self.root, 'features', type + '_weak_label.pkl')
        elif type in ['val', 'test']:
            label_file = os.path.join(self.root, 'features', type + '_strong_label.pkl')
        else:
            print('ERROR data type in load_urbansed()!')
            return
        with open(data_file, 'rb') as f:
            x_data = pkl.load(f)
        with open(label_file, 'rb') as f:
            y_data = pkl.load(f)

        return x_data, y_data

    def seq_norm(self, s, max_secs=10):
        while (len(s) < self.sr * max_secs):
            s = np.concatenate([s, s[: self.sr * max_secs - len(s)]])
        return s[:self.sr * max_secs]

    def get_feats(self, audio_path, type, outpath):
        fw = self.frame_width
        fs = self.frame_step
        if isinstance(audio_path, list):
            files = []
            for path in audio_path:
                files += self.get_absolute_path(path)
        else:
            files = self.get_absolute_path(audio_path)
        feat_dict = {'filenames':files, 'data':[]}
        files_bar = tqdm(range(len(files)))
        nframe = int(np.floor((self.clip_len * self.sr - fw) / fs) + 1)
        n_mels = self.n_mels
        feats = np.zeros((len(files), n_mels, nframe), np.float32)
        error_files = []
        for i in files_bar:
            file = files[i]
            files_bar.set_description('FBanks extraction {}/{}'.format(i, len(files)))
            try:
                s, _ = lrs.load(file, sr=self.sr)
                s = self.seq_norm(s)
                s = np.array(s).reshape((-1, 1))
                s = np.concatenate((s[0:1], s[1:] - 0.97 * s[:-1]))
                melspec = lrs.feature.melspectrogram(y=s.squeeze(), sr=self.sr, n_fft=fw,
                                                    hop_length=fs, n_mels=n_mels,
                                                    win_length=fw,
                                                    center=False,
                                                    window="hamming",
                                                    #  power=1.0
                                                    )
                logmelspec = lrs.power_to_db(melspec)
                feats[i] = logmelspec  
            except:
                error_files.append(file)
                continue
        feat_dict['data'] = feats

        with open(os.path.join(outpath, type + '.pkl'), 'wb') as f:
            pkl.dump(feat_dict, f)

    def load_dcase18(self, type):
        data_file = os.path.join(self.root, 'features', type + '.pkl')
        if type in ['training']:
            label_file = os.path.join(self.root, 'features', type + '_weak_label.pkl')
        elif type in ['testing', 'evaluation']:
            label_file = os.path.join(self.root, 'features', type + '_strong_label.pkl')
        else:
            print('ERROR data type in load_dcase18()!')
            return
        with open(data_file, 'rb') as f:
            x_data = pkl.load(f)
        with open(label_file, 'rb') as f:
            y_data = pkl.load(f)
        return x_data, y_data
    
    def load_dcase19(self, type):
        data_file = os.path.join(self.root, 'features', type + '.pkl')
        if type in ['training']:
            label_file = os.path.join(self.root, 'features', type + '_weak_label.pkl')
            synthetic_file = os.path.join(self.root, 'features', 'synthetic_strong_label.pkl')
            with open(synthetic_file, 'rb') as f:
                synthetic_label = pkl.load(f)
                synthetic_weak_label = (synthetic_label['labels'].sum(1) > 1).astype(np.float32)
                synthetic_weak_dict = {'filenames': synthetic_label['filenames'], 'labels':synthetic_weak_label}
        elif type in ['testing', 'evaluation']:
            label_file = os.path.join(self.root, 'features', type + '_strong_label.pkl')
        else:
            print('ERROR data type in load_dcase18()!')
            return
        with open(data_file, 'rb') as f:
            x_data = pkl.load(f)
        with open(label_file, 'rb') as f:
            y_data = pkl.load(f)
            if type == 'training':
                y_data['filenames'] += synthetic_weak_dict['filenames']
                y_data['labels'] = np.concatenate([y_data['labels'], synthetic_weak_dict['labels']])
        return x_data, y_data

    def load_anuraset(self, type_):
        metadata_path = os.path.join(self.root, "metadata.csv")
        metadata = pd.read_csv(metadata_path)
        subset = "train" if type_ == "training" else "test"
        subset_meta = metadata[metadata["subset"] == subset]
        class_columns = list(metadata.columns[metadata.columns.get_loc("subset") + 1 :])
        if mc.TARGET_SPECIES:
            class_columns = [col for col in class_columns if col in mc.TARGET_SPECIES]

        strong_csv = mc.STRONG_LABELS_578 if "578" in mc.FNJV_ROOT else mc.STRONG_LABELS_458
        strong_events = None
        if os.path.exists(strong_csv):
            strong_df = pd.read_csv(strong_csv)
            if mc.STRONG_LABEL_LEVELS:
                strong_df = strong_df[strong_df["level"].isin(mc.STRONG_LABEL_LEVELS)]
            strong_df["file_stem"] = strong_df["file_name"].str.replace(".txt", "", regex=False)
            strong_events = {}
            for _, row in strong_df.iterrows():
                strong_events.setdefault(row["file_stem"], []).append(
                    {
                        "start": float(row["start_second"]),
                        "end": float(row["end_second"]),
                        "label": str(row["label"]),
                    }
                )

        data = []
        labels = []
        audio_files = []
        fw = self.frame_width
        fs = self.frame_step
        nframe = int(np.floor((self.clip_len * self.sr - fw) / fs) + 1)
        for _, row in subset_meta.groupby(["site", "fname"]).first().iterrows():
            fname = row["fname"]
            site = row["site"]
            file_labels = subset_meta[subset_meta["fname"] == fname][class_columns].max().values.astype(np.float32)
            audio_path = os.path.join(self.root, site, f"{fname}.wav")
            try:
                s, _ = lrs.load(audio_path, sr=self.sr)
            except Exception:
                continue
            s = self.seq_norm(s, max_secs=self.clip_len)
            s = np.array(s).reshape((-1, 1))
            s = np.concatenate((s[0:1], s[1:] - 0.97 * s[:-1]))
            melspec = lrs.feature.melspectrogram(
                y=s.squeeze(),
                sr=self.sr,
                n_fft=fw,
                hop_length=fs,
                n_mels=self.n_mels,
                win_length=fw,
                center=False,
                window="hamming",
            )
            logmelspec = lrs.power_to_db(melspec).T
            if logmelspec.shape[0] < nframe:
                pad = np.zeros((nframe - logmelspec.shape[0], self.n_mels), dtype=logmelspec.dtype)
                logmelspec = np.concatenate([logmelspec, pad], axis=0)
            elif logmelspec.shape[0] > nframe:
                logmelspec = logmelspec[:nframe]
            data.append(self.standardization(logmelspec))
            audio_files.append(fname)
            if strong_events is not None:
                events = strong_events.get(fname, [])
                strong = np.zeros((nframe, len(class_columns)), dtype=np.float32)
                frames_per_sec = nframe / self.clip_len
                for event in events:
                    if event["label"] not in class_columns:
                        continue
                    class_idx = class_columns.index(event["label"])
                    start_frame = int(np.floor(event["start"] * frames_per_sec))
                    end_frame = int(np.ceil(event["end"] * frames_per_sec))
                    start_frame = max(0, min(nframe, start_frame))
                    end_frame = max(0, min(nframe, end_frame))
                    if end_frame > start_frame:
                        strong[start_frame:end_frame, class_idx] = 1.0
                labels.append(strong)
            else:
                labels.append(file_labels)
        return data, labels, audio_files

    def ann_to_labels_dcase17(self, ann_file, type):
        root = '../sed_data/dcase/sound_event_list_17_classes.txt'
        event_list = []
        audio_files = sorted(os.listdir('../sed_data/dcase/audio/' + type))
        with open(root, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                event = line.split('\t')[-1].strip('\n')
                event_list.append(event)
        files = open(ann_file, 'r')
        fw = self.frame_width
        fs = self.frame_step
        nframe = int(np.floor((self.clip_len * self.sr - fw) / fs) + 1)
        loc_labels = np.zeros((len(audio_files), nframe, len(event_list)), dtype=np.float)

        for line in files.readlines():
            item = line.strip('\n').split('\t')
            hash_id = audio_files.index('Y' + item[0])
            onset = float(item[1])
            offset = float(item[2])
            label = event_list.index(item[3])
            step_sec = self.frame_step / self.sr
            on_frame = int(np.floor(onset / step_sec))
            off_frame = min(int(offset / step_sec), nframe)
            loc_labels[hash_id, on_frame:off_frame, label] = 1.0

        label_dict = {'audio_files': audio_files, 'loc_labels': loc_labels}
        pkl.dump(label_dict, file=open('../sed_data/dcase/features_base/' + type + '_label.pkl', 'wb'))

    def get_weak_labels_dcase(self, ann_file, type, outpath):
        event_list = self.config['data']['labels']
        if isinstance(ann_file, list):
            labels_df = pd.DataFrame()
            for path in ann_file:
                labels_df = pd.concat([labels_df, pd.read_csv(path, sep='\t')])
        else:
            labels_df = pd.read_csv(ann_file, sep='\t')
        file_lists = sorted(list(labels_df.values[:, 0]))
        tag_labels = np.zeros((len(file_lists), len(event_list)), np.float32)
        for line in labels_df.values:
            filename = line[0]
            file_id = file_lists.index(filename)
            # file_lists[file_id] = os.path.join(root, filename)
            activated_events = line[-1].split(',')
            for e in activated_events:
                idx = event_list.index(e)
                tag_labels[file_id, idx] = 1.
        weak_dict = {'filenames': file_lists, 'labels': tag_labels}
        pkl.dump(weak_dict, file=open(os.path.join(outpath,  type + '_weak_label.pkl'), 'wb'))

    def get_strong_labels_dcase(self, ann_file, type, outpath):
        event_list = self.config['data']['labels']
        if isinstance(ann_file, list):
            labels_df = pd.DataFrame()
            for path in ann_file:
                labels_df = pd.concat([labels_df, pd.read_csv(path, sep='\s+').convert_dtypes()])
        else:
            labels_df = pd.read_csv(ann_file, sep='\s+').convert_dtypes()
        fw = self.frame_width
        fs = self.frame_step
        nframe = int(np.floor((self.clip_len * self.sr - fw) / fs) + 1)
        filelists = sorted(list(set(labels_df['filename'])))
        loc_labels = np.zeros((len(filelists), nframe, len(event_list)), np.float32)
        for line in labels_df.values:
            filename = line[0]
            file_id = filelists.index(filename)
            try:
                onset = float(line[1])
                offset = float(line[2])
            except:
                continue
            event_id = event_list.index(line[3])
            step_sec = self.frame_step / self.sr
            on_frame = int(np.floor(onset / step_sec))
            off_frame = min(int(offset / step_sec), nframe)
            loc_labels[file_id, on_frame:off_frame, event_id] = 1.0
        strong_dict = {'filenames': filelists, 'labels': loc_labels}
        pkl.dump(strong_dict, file=open(os.path.join(outpath,  type + '_strong_label.pkl'), 'wb'))
        
    def get_train_label(self, ann_file):
        root = '../sed_data/dcase/sound_event_list_17_classes.txt'
        event_list = []
        with open(root, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                event = line.split('\t')[-1].strip('\n')
                event_list.append(event)
        audio_files = sorted(os.listdir('../sed_data/dcase/audio/training'))
        loc_labels = np.zeros((len(audio_files), len(event_list)), np.float16)
        files = open(ann_file, 'r')
        for line in files.readlines():
            item = line.strip('\n').split('\t')
            hash_id = audio_files.index('Y' + item[0])
            label = event_list.index(item[3])
            loc_labels[hash_id, label] = 1

        label_dict = {'audio_files': audio_files, 'loc_labels': loc_labels}
        pkl.dump(label_dict, file=open('../sed_data/dcase/features_base/training_label.pkl', 'wb'))

# dh = DataHandler()
# dh.load_urbansed(type='train')
# dh.get_weak_labels_urbansed(ann_file='urban_sed_train_weak.tsv', type='train')
# dh.get_weak_labels_urbansed(ann_file='urban_sed_validation_weak.tsv', type='val')
# dh.get_strong_labels_urbansed(ann_file='urban_sed_test_strong.tsv', type='test')


#
# outpath='../sed_data/dcase/features/'
# train_files = dh.get_absolute_path('/data0/gaolj/sed_data/dcase/audio/training')
# dh.get_feats(train_files, type='training', outpath=outpath)
# test_files = dh.get_absolute_path('../sed_data/dcase/audio/test')
# dh.get_feats(test_files, type='test', outpath=outpath)
# evaluation_files = dh.get_absolute_path('../sed_data/dcase/audio/evaluation')
# dh.get_feats(evaluation_files, type='evaluation', outpath=outpath)
#
# dh.ann_to_labels(ann_file='../sed_data/dcase/groundtruth_strong_label_testing_set.txt', type='testing')
# dh.ann_to_labels(ann_file='../sed_data/dcase/groundtruth_strong_label_evaluation_set.txt', type='evaluation')
# dh.get_train_label(ann_file='../sed_data/dcase/groundtruth_weak_label_training_set.txt')
