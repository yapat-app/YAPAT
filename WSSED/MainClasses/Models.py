import math
import torch
from torch import nn
from MainClasses.MILPooling import MILPooling

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in: (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

class ModelBase(nn.Module):
    def __init__(self, n_classes, pool_style, seq_len=None):
        super().__init__()
        self.pool_style = pool_style
        self.pool = MILPooling(n_classes=n_classes, seq_len=seq_len).get_pool(pool_style)
    def pool_with_mask(self, y_frames, mask=None):
        if mask is None:
            return self.pool(y_frames)
        try:
            return self.pool(y_frames, mask)
        except TypeError:
            return self.pool(y_frames)
    def unsample(self, y_frames, org_seq_len):
        y_frames = torch.nn.functional.interpolate(
            y_frames.transpose(1, 2),
            org_seq_len,
            mode='linear',
            align_corners=False).transpose(1, 2)
        return y_frames

class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, dilation=1, padding=1):
        super().__init__()
        if padding == 1:
            padding_value = self.padding_same_value(kernel_size, dilation)
        else:
            padding_value = 0
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding_value,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))
        self.init_weights()

    def padding_same_value(self, kernel, dilation):
        return (kernel // 2) * dilation

    def init_weights(self):
        for layer in self.block:
            if hasattr(layer, 'weights'):
                try:
                    init_layer(layer)
                except:
                    init_bn(layer)

    def forward(self, x):
        return self.block(x)

class Block1D(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, dilation=1, drop_rate=0.2, padding=0, stride=1):
        super(Block1D, self).__init__()
        if padding == 1:
            padding_value = self.padding_same_value(kernel_size, dilation)
        else:
            padding_value = 0
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=stride,
                      padding=padding_value, dilation=dilation, bias=False),
            nn.BatchNorm1d(num_features=filters),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        self.init_weigths()

    def init_weigths(self):
        for layer in self.block:
            if hasattr(layer, 'weights'):
                try:
                    init_layer(layer)
                except:
                    init_bn(layer)

    def padding_same_value(self, kernel, dilation):
        return (kernel // 2) * dilation

    def forward(self, inputs):
        x = self.block(inputs)
        return x

class CRNN1D(nn.Module):
    def __init__(self, n_classes=10):
        super(CRNN1D, self).__init__()
        self.conv_block1 = Block1D(64, 128, kernel_size=5, drop_rate=0.3, padding=1)
        self.conv_block2 = Block1D(128, 128, kernel_size=5, drop_rate=0.2, padding=1)
        self.conv_block3 = Block1D(128, 256, kernel_size=7, drop_rate=0.2, padding=1)
        self.conv_block4 = Block1D(256, 512, kernel_size=7, drop_rate=0.2, dilation=2, padding=1)
        self.conv_block5 = Block1D(512, 512, kernel_size=11, drop_rate=0.2, dilation=3, padding=1)
        self.conv_block6 = Block1D(512, 512, kernel_size=13, drop_rate=0.2, dilation=3, padding=1)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=3,
            batch_first=True
        )
        self.out = nn.Linear(256, n_classes)
        self.init_weights()

    def init_weights(self):
        init_gru(self.rnn)
        init_layer(self.out)

    def forward(self, input):
        x = self.conv_block1(input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x, _ = self.rnn(x.transpose(1, 2), None)
        y_frame = torch.sigmoid(self.out(x)).clamp(1e-7, 0.99)
        return y_frame
        
class CRNN(nn.Module):
    def __init__(self, outputdim):
        super(CRNN, self).__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.Dropout(0.3),
            nn.MaxPool2d((2, 1)),
            Block2D(32, 128),
            nn.Dropout(0.3),
            nn.MaxPool2d((2, 1)),
            Block2D(128, 128),
            nn.Dropout(0.3),
            nn.MaxPool2d((4, 1)),
            Block2D(128, 128),
            nn.Dropout(0.3),
            Block2D(128, 128),
            nn.Dropout(0.3),
            nn.AvgPool2d((4, 1))
        )
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Linear(256, outputdim)
        self.init_weights()

    def init_weights(self):
        init_gru(self.rnn)
        init_layer(self.out)

    def forward(self, input):
        x = self.features(input.unsqueeze(1)).squeeze(2)
        x, _ = self.rnn(x.transpose(1, 2), None)
        y_frame = torch.sigmoid(self.out(x)).clamp(1e-7, 0.99)
        return y_frame


class TALNet(ModelBase):
    def __init__(self, n_classes, pool_style, seq_len):
        super().__init__(n_classes, pool_style, seq_len)
        self.name = 'talnet_' + pool_style
        self.cnn = nn.Sequential(
            Block2D(1, 32, kernel_size=5),
            nn.MaxPool2d((1, 2)),
            Block2D(32, 64, kernel_size=5),
            nn.MaxPool2d((1, 2)),
            Block2D(64, 128, kernel_size=5),
            nn.MaxPool2d((1, 2)),
        )
        self.rnn = nn.GRU(
            1024,
            100,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.out = nn.Linear(200, n_classes)
        self.init_weights()

    def init_weights(self):
        init_gru(self.rnn)
        init_layer(self.out)

    def forward(self, inputs, upsample=False, mask=None, **_kwargs):
        x = self.cnn(inputs.unsqueeze(1)).transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x, _ = self.rnn(x)
        y_frames = torch.sigmoid(self.out(x))
        y_clip = self.pool_with_mask(y_frames, mask)
        if upsample:
            y_frames = self.unsample(y_frames, inputs.shape[1])
        return y_clip, y_frames


class Baseline(ModelBase):
    def __init__(self, n_classes, pool_style, seq_len):
        super().__init__(n_classes, pool_style, seq_len)
        self.seq_len = 240
        self.crnn = CRNN1D(n_classes)
        self.name = 'baseline_' + pool_style

    def forward(self, inputs, upsample=False, mask=None, **_kwargs):
        y_frames = self.crnn(inputs.transpose(1, 2))
        y_clip = self.pool_with_mask(y_frames, mask)
        if upsample:
            y_frames = self.unsample(y_frames, inputs.shape[1])
        return y_clip, y_frames

class CDur(ModelBase):
    """
    Implementation of the paper:

    Towards Duration Robust Weakly Supervised Sound Event Detection

    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9335265&tag=1
    """
    def __init__(self, n_classes, pool_style, seq_len):
        super().__init__(n_classes, pool_style, seq_len)
        self.name = 'cdur_' + pool_style
        self.cnn = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (1, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )

        self.gru = nn.GRU(128,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.outputlayer = nn.Linear(256, n_classes)
        self.init_weights()

    def init_weights(self):
        init_gru(self.gru)
        init_layer(self.outputlayer)
        
    def forward(self, x, upsample=True, mask=None, **_kwargs):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        y_frames = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        y_clip = self.pool_with_mask(y_frames, mask)
        if upsample:
            y_frames = self.unsample(y_frames, time)
        return y_clip.clamp(1e-7, 1.), y_frames.clamp(1e-7, 1.)
