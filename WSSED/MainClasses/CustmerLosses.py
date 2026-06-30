import torch
import torch.nn as nn


class DKLLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss
    """

    def __init__(self):
        super(DKLLoss, self).__init__()

    def forward(self, z_stars, alphas, gaussian_units):
        mu = gaussian_units[0]
        log_var = gaussian_units[1]
        mu_star = alphas * mu.unsqueeze(-1)
        log_var_star = alphas * log_var.unsqueeze(-1)
        kl = - 0.5 * torch.sum(1 + log_var_star - mu_star.pow(2) - log_var_star.exp(), dim=1)
        return kl


class DisentLoss(nn.Module):
    """
    Proposed supervised disentangling loss
    """

    def __init__(self, K, beta):
        super(DisentLoss, self).__init__()
        self.detector_loss = nn.BCELoss(reduction='none')
        self.dkl = DKLLoss()
        self.K = K
        self.beta = beta

    def forward(self, inputs, targets):
        detector_out = inputs[0]
        z_star = inputs[1]
        alphas = inputs[2]
        gaussian_units = inputs[3]
        dkl = self.dkl(z_star, alphas, gaussian_units)
        detector_l = self.detector_loss(detector_out, targets)
        disent_loss = (detector_l + self.beta * dkl).mean()
        disent_loss = torch.mean(disent_loss, dim=0)
        return disent_loss, (detector_l.mean().cpu(), dkl.mean().cpu())
