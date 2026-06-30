import numpy as np
import librosa as lrs
import pickle as pkl
from tqdm import tqdm


def mel_bank(N, n_fft, sr):
    '''
    定义梅尔滤波组
    :param N: 滤波器数量
    :param n_fft: FFT点数
    :param sr: 采样率
    :return: bank - 梅尔滤波带系数矩阵，NxM
                    M - 正频部分的下标最大值
    '''
    M = int(np.floor(n_fft / 2))
    f = np.arange(0, M) / M * sr / 2
    f_mel = 2595*np.log10(1 + f / 700)
    delta = f_mel[M - 1] / (N + 1)

    bank = np.zeros((N, M))
    center = 0
    for n in range(N):
        center += delta
        for m in range(M):
            if f_mel[m] >= center - delta and f_mel[m] <= center + delta:
                bank[n, m] = 1 - np.abs(f_mel[m] - center) / delta
            else:
                bank[n, m] = 0

    return bank, M


def enframe(signal, frame_width, frame_step):
    '''
    分帧
    :param signal: 语音信号
    :param frame_step: 帧移
    :param frame_width: 帧长
    :return: frames - 信号帧； nframes - 帧数
    '''
    sample_width = int(frame_width)
    step = int(frame_step )
    slength = len(signal)
    nframes = int(np.floor((slength - sample_width) / step) + 1)
    indices = np.tile(np.arange(0, sample_width), (nframes, 1)) + \
              np.tile(np.arange(0, nframes * step, step), (sample_width, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = signal[indices]
    return frames, nframes


def f_bank(signal, frame_width, frame_step, M, mel_bank):
    '''
    定义FBank特征
    :param signal: 输入信号
    :param frame_width: 帧长
    :param frame_step: 帧移
    :param M: 正频下标最大值
    :param mel_bank: Mel滤波器
    :return: frames - 信号帧，帧数x帧长的矩阵
    '''
    signal = np.array(signal).reshape((-1, 1))
    signal = np.concatenate((signal[0:1], signal[1:] - 0.7 * signal[:-1]))
    hanming = 0.54 - 0.46 * np.cos(2 * 3.141593 * np.arange(0, frame_width)
                                                    / (frame_width - 1)).reshape(-1, 1)

    frames, nframes = enframe(signal, frame_width, frame_step)
    f_bank_feat = np.zeros((nframes, mel_bank.shape[0]))
    EPS = 1e-10

    for n in range(nframes):
        # 1.加窗
        # 2.fft变换
        # 3.取正频部分
        # 4.振幅 = 实部和虚部平方和
        frame = frames[n, :] * hanming
        f = np.fft.fft(frame.T).squeeze(0)
        f = f[0:M]
        f = np.float32(f * np.conj(f))
        f_bank_feat[n, :] = np.log(np.matmul(mel_bank, f.reshape(-1, 1)).squeeze(-1) + EPS)

    return f_bank_feat


def get_fbanks(files, n_mels, frame_width, frame_step, sr, out_path=None):
    """
    批量生成FBank特征，并写入文件，默认命名为：信号文件名.pkl
    :param files: 文件列表
    :param frame_width: 帧长
    :param frame_step: 帧移
    :param M: 正频下标最大值
    :param out_path: 输出路径
    """
    bank, M = mel_bank(N=n_mels, n_fft=frame_width, sr=sr)
    files_bar = tqdm(range(len(files)))
    for i in files_bar:
        file = files[i]
        files_bar.set_description('FBanks extraction {}/{}'.format(i, len(files)))
        s, _ = lrs.load(file, sr=sr)
        feats = f_bank(s, frame_width, frame_step, M, bank)
        if feats.max() - feats.min() == 0:
            break
        if out_path is not None:
            with open(out_path + file.split('.wav')[0].split('/')[-1] + '.pkl', 'wb') as f:
                pkl.dump(feats, f)
        else:
            return feats