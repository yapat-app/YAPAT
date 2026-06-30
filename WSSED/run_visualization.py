from MainClasses.Learner import Learner
from MainClasses.Models import CDur
from torch.utils.data import DataLoader
from MainClasses.TorchDataset import DCASE2017
import pandas as pd
from MainClasses.Evaluator import Evaluator
from argparse import ArgumentParser
from torch import nn
import torch
import os
import json
pd.set_option('precision', 4)

DATASET = 'DCASE2017'   
def args_setup():
    parser = ArgumentParser()
    # NOTE: parameters below are not supposed to change.
    parser.add_argument("-dt", "--dataset", type=str, default=DATASET)
    parser.add_argument('-o', "--result_path", type=str, default='results/' + DATASET)
    parser.add_argument('-c', "--config", default=json.load(open('./MainClasses/config.json', 'r'))[DATASET])

    # Parameters below are changeable.
    parser.add_argument('-w', '--num_workers', type=int, default=0)
    parser.add_argument('-d', "--feature_dim", type=int, default=64)
    parser.add_argument('-b', "--batch_size", type=int, default=196)
    parser.add_argument('-e', "--epoch", type=int, default=200)
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.001)
    parser.add_argument('-k', "--num_events", type=int, default=17)
    parser.add_argument('-gpu', "--gpu_device", type=str, default='0')
    parser.add_argument('--eval_only', action='store_true', default=False)
    """
    TODO: Change the pool style below in [max_pool, avg_pool, attention_pool, linear_pool, exp_pool, auto_pool, power_pool, hi_pool, hi_pool_plus]
    """
    parser.add_argument('-p', "--pooling", type=str, default='hi_pool')
    return parser


if __name__ == '__main__':
    args = args_setup()
    options = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_device
    batch_size = options.batch_size

    tagging_loss = nn.BCELoss()
    f1_tag_list, f1_loc_list = [], []
    model_hi_pool = CDur(pool_style='hi_pool', n_classes=17, seq_len=240).cuda()
    model_max_pool = CDur(pool_style='max_pool', n_classes=17, seq_len=240).cuda()
    model_avg_pool = CDur(pool_style='avg_pool', n_classes=17, seq_len=240).cuda()

    # 2.Load data.
    eval_dataset = DCASE2017(options, type_='evaluation')
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=options.num_workers,
                             drop_last=False)

    eval = Evaluator(options, data_loader=eval_loader)

    eval.loc_plot([model_hi_pool, model_max_pool, model_avg_pool])