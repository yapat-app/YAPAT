import numpy as np
from sklearn.metrics import confusion_matrix


def r_square(inputs, targets):
    y_mean = np.mean(targets, axis=-1, keepdims=True)
    SSE = np.sum(np.square(targets - inputs), axis=-1)
    SST = np.sum(np.square(targets - y_mean), axis=-1)
    R2 = 1 - SSE / (SST + 1e-10)
    return R2.mean()

# def tagging_metrics(inputs, targets, K):

def segment_metrics(inputs, targets, K):
    from scipy.signal import medfilt

    def frame_to_segments(event_label):
        total_s = int(np.ceil(len(event_label) / 50))  # 0.02s per frame, 50 frames = 1.0s
        segments = np.zeros((total_s, K), np.int8)

        for i in range(total_s):
            if (i + 1) * 50 > len(event_label):
                res = np.zeros(((i + 1) * 50 - len(event_label), K))
                event_label = np.concatenate((event_label, res))
            per_s = event_label[i * 50:(i + 1) * 50]
            for j in range(K):
                if sum(per_s[:, j]) > 0:
                    segments[i, j] = 1
        return segments

    def count_factors(y_pers, y_hat_pers, overall):
        Ntp = sum(y_hat_pers + y_pers > 1)
        Ntn = sum(y_hat_pers + y_pers == 0)
        Nfp = sum(y_hat_pers - y_pers > 0)
        Nfn = sum(y_pers - y_hat_pers > 0)

        Nref = sum(y_pers)
        Nsys = sum(y_hat_pers)

        S = min(Nref, Nsys) - Ntp
        D = max(0, Nref - Nsys)
        I = max(0, Nsys - Nref)

        overall['Ntp'] += Ntp
        overall['Ntn'] += Ntn
        overall['Nfp'] += Nfp
        overall['Nfn'] += Nfn
        overall['Nref'] += Nref
        overall['Nsys'] += Nsys
        overall['S'] += S
        overall['D'] += D
        overall['I'] += I

    y_hat = inputs
    y_hat_filted = np.zeros(y_hat.shape, np.int8)
    overall = {
        'Ntp': 0.0,
        'Ntn': 0.0,
        'Nfp': 0.0,
        'Nfn': 0.0,
        'Nref': 0.0,
        'Nsys': 0.0,
        'ER': 0.0,
        'S': 0.0,
        'D': 0.0,
        'I': 0.0,
    }
    for i in range(K):
        activity_array = y_hat[:, i] > 0.5
        event_label = medfilt(volume=activity_array, kernel_size=27)  # 0.54s做滤波
        event_label = np.array(event_label, np.int8)
        y_hat_filted[:, i] = event_label

    y_hat_segments = frame_to_segments(y_hat_filted)
    y_segments = frame_to_segments(targets)
    for i in range(len(y_hat_segments)):
        count_factors(y_segments[i], y_hat_segments[i], overall)
    # calculate F1
    eps = np.spacing(1)

    precision = overall['Ntp'] / float(overall['Nsys'] + eps)
    recall = overall['Ntp'] / float(overall['Nref'] + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)
    # calculate error
    substitution_rate = float(overall['S'] / (overall['Nref'] + eps))
    deletion_rate = float(overall['D'] / (overall['Nref'] + eps))
    insertion_rate = float(overall['I'] / (overall['Nref'] + eps))
    error_rate = float(substitution_rate + deletion_rate + insertion_rate)
    return f1_score, error_rate


def binary_accurate(inputs, targets):
    correct = np.array(inputs == targets, np.int8)
    acc = correct.mean(axis=(0, 1))
    return acc

