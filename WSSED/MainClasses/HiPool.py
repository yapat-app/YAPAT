import torch
from torch import nn

def linear_softmax(yi):
    return torch.sum(yi ** 2, dim=1) / torch.sum(yi, dim=1)


class HiPoolFamily(nn.Module):
    def __init__(self, seq_len, n_classes):
        super(HiPoolFamily, self).__init__()
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.n_star = []
        for n in range(1, seq_len + 1):
            if seq_len % n == 0:
                self.n_star.append(n)

    def init_layer(self, layer):
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def pooling(self, yi, r):
        bin_len = self.seq_len // r
        yi_bin = torch.max_pool1d(yi, kernel_size=bin_len, stride=bin_len)
        y = yi_bin.mean(-1)
        return y

    def forward(self, *input):
        return

class HiPool(HiPoolFamily):
    def __init__(self, seq_len, n_classes):
        super(HiPool, self).__init__(seq_len, n_classes)
        self.tag = True
        w = torch.ones(n_classes, len(self.n_star))
        w[:, -1] += 0.01
        self.w = nn.Parameter(w, requires_grad=True)

    def forward(self, y_frame, mask=None):
        if mask is not None:
            y_frame = y_frame * mask.unsqueeze(-1)
        alphas = self.w.softmax(-1)
        if self.training:
            y_clip = torch.zeros((len(y_frame), self.n_classes, len(self.n_star)), requires_grad=True).cuda()
            for k in range(self.n_classes):
                for i in range(len(self.n_star)):
                    y_clip[:, k, i] = (alphas[k, i] * self.pooling(y_frame[:, :, k].unsqueeze(1), self.n_star[i])).squeeze()
            y_clip = y_clip.sum(-1)
            self.tag = True
        else:
            if self.tag:
                print(torch.tensor(self.n_star)[alphas.argmax(-1)])
                # print(alphas)
                self.tag = False
            y_clip = torch.zeros((len(y_frame), self.n_classes)).cuda()
            for k in range(self.n_classes):
                r = self.n_star[int(alphas[k].argmax(-1))]
                y_clip[:, k] = self.pooling(y_frame[:, :, k].unsqueeze(1), r).squeeze()
        return y_clip


class HiPoolPlus(HiPoolFamily):
    def __init__(self, seq_len, n_classes):
        super(HiPoolPlus, self).__init__(seq_len, n_classes)
        self.w = nn.Parameter(torch.ones(n_classes, len(self.n_star)), requires_grad=True)
        self.tag = True

    def pooling(self, yi, r):
        bin_len = self.seq_len // r
        yi_bin = torch.max_pool1d(yi, kernel_size=bin_len, stride=bin_len)
        y = torch.sum(yi_bin * torch.exp(yi_bin), dim=-1) / torch.sum(torch.exp(yi_bin), dim=-1)
        return y

    def forward(self, y_frame, mask=None):
        if mask is not None:
            y_frame = y_frame * mask.unsqueeze(-1)
        alphas = self.w.softmax(-1)
        if self.training:
            self.tag = False
            y_clip = torch.zeros((len(y_frame), self.n_classes, len(self.n_star)), requires_grad=True).cuda()
            for k in range(self.n_classes):
                for i in range(len(self.n_star)):
                    y_clip[:, k, i] = (alphas[k, i] * self.pooling(y_frame[:, :, k].unsqueeze(1), self.n_star[i])).squeeze()
            y_clip = y_clip.sum(-1)
        else:

            if self.tag:
                print("Adaptive R with Seq {} : {}", self.seq_len, alphas.argmax(-1))
                self.tag = False
            y_clip = torch.zeros((len(y_frame), self.n_classes)).cuda()
            for k in range(self.n_classes):
                r = self.n_star[int(alphas[k].argmax(-1))]
                y_clip[:, k] = self.pooling(y_frame[:, :, k].unsqueeze(1), r).squeeze()
        return y_clip


class HiPoolFixed(HiPoolFamily):
    def __init__(self, seq_len, n_classes):
        super(HiPoolFixed, self).__init__(seq_len, n_classes)
    def set_R(self, R):
        self.R = R
    def forward(self, y_frame, mask=None):
        if mask is not None:
            y_frame = y_frame * mask.unsqueeze(-1)
        y_clip = torch.zeros((len(y_frame), self.n_classes)).cuda()
        for k in range(self.n_classes):
            y_clip[:, k] = self.pooling(y_frame[:, :, k].unsqueeze(1), self.R).squeeze()
        return y_clip
