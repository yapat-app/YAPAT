import numpy as np
from scipy.signal import medfilt


class Metrics:
    def __init__(self, K=17, seg_len=24):
        self.tp = np.zeros(K)
        self.tn = np.zeros(K)
        self.fp = np.zeros(K)
        self.fn = np.zeros(K)
        self.er = np.zeros(K)
        self.seg_len = seg_len

    def initialization(self):
        self.__init__()

    def inference_by_th(self, inputs, threshold=0.5):
        idx = np.where(inputs > threshold)
        out = np.zeros(shape=inputs.shape, dtype=np.float16)
        out[idx] = 1.0
        return out

    def _label_smooth(self, yi, kernel_size=11):
        K = yi.shape[-1]
        yi_smoothed = np.zeros(yi.shape, np.int8)
        for i in range(K):
            y_hat = medfilt(volume=yi[:, :, i], kernel_size=kernel_size)  # 0.54s做滤波
            y_hat = np.array(y_hat, np.int8)
            yi_smoothed[:, :, i] = y_hat
        return yi_smoothed

    def frame2seg(self, event_label):
        total_s = int(np.ceil(len(event_label) / self.seg_len))  # 0.02s per frame, 50 frames = 1.0s
        K = len(event_label[0])
        segments = np.zeros((total_s, K), np.int8)

        for i in range(total_s):
            if (i + 1) * self.seg_len > len(event_label):
                res = np.zeros(((i + 1) * self.seg_len - len(event_label), K))
                event_label = np.concatenate((event_label, res))

            per_s = event_label[i * self.seg_len:(i + 1) * self.seg_len]
            for j in range(K):
                if sum(per_s[:, j]) > 0:
                    segments[i, j] = 1
        return segments

    def count(self, inputs, targets, if_seg=False):
        inputs = self.inference_by_th(inputs, threshold=0.5)
        inputs = self._label_smooth(inputs)

        if if_seg:
            inputs_seg, targets_seg = [], []
            for i in range(len(inputs)):
                inputs_seg.append(self.frame2seg(inputs[i]))
                targets_seg.append(self.frame2seg(targets[i]))
            inputs = np.array(inputs_seg)
            targets = np.array(targets_seg)

            for b in range(len(targets)):
                self.tp += sum(inputs[b] + targets[b] > 1)
                self.tn += sum(inputs[b] + targets[b] == 0)
                self.fp += sum(inputs[b] - targets[b] > 0)
                self.fn += sum(targets[b] - inputs[b] > 0)
        else:
            self.tp += sum(inputs + targets > 1)
            self.tn += sum(inputs + targets == 0)
            self.fp += sum(inputs - targets > 0)
            self.fn += sum(targets - inputs > 0)

    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-10)

    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-10)

    def f1_score(self):
        return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall() + 1e-10)

    def error_rate(self):
        for k in range(len(self.er)):
            S = min(self.tp[k] + self.fn[k], self.tp[k] + self.fp[k]) - self.tp[k]
            D = max(0, self.fn[k] - self.fp[k])
            I = max(0, self.fp[k] - self.fn[k])
            substitution_rate = float(S / (self.tp[k] + self.fn[k] + 1e-10))
            deletion_rate = float(D / (self.tp[k] + self.fn[k] + 1e-10))
            insertion_rate = float(I / (self.tp[k] + self.fn[k] + 1e-10))
            er = float(substitution_rate + deletion_rate + insertion_rate)
            self.er[k] = er
        return self.er
