from MainClasses.Learner import Learner
from MainClasses.Models import CDur, Baseline, TALNet
import torch
from MainClasses.MILPooling import MILPooling
from MainClasses.TorchDataset import DCASE2017
import pandas as pd
from argparse import ArgumentParser
from torch import nn
import os
import json
import pickle as pkl
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
    parser.add_argument('--eval_only', action='store_true', default=True)
    parser.add_argument('-p', "--pooling", type=str, default='hi_pool_fixed')
    return parser


def write_metrics(checkpoint='f1_tag', dataset=DATASET):
    # evaluation, checkpoint = F1 on tagging
    print('============Evaluation==============')
    eval_data = 'evaluation'
    tag_f1, loc_f1 = learner.evaluation(eval_data, cp=checkpoint, online=False, avg=False)
    return tag_f1, loc_f1


if __name__ == '__main__':
    args = args_setup()
    options = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_device
    batch_size = options.batch_size
    epoch = options.epoch
    lr = options.learning_rate
    event_labels = options.config['data']['labels']
    pool_style = options.pooling
    tagging_loss = nn.BCELoss()

    train_dataset = DCASE2017(options, type_='training')
    test_dataset = DCASE2017(options, type_='testing')
    eval_dataset = DCASE2017(options, type_='evaluation')
    results = []

    for R in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 240, 'star']:
        if R == 1:
            model = Baseline(pool_style='max_pool', n_classes=17, seq_len=240).cuda()
        elif R == 240:
            model = Baseline(pool_style='avg_pool', n_classes=17, seq_len=240).cuda()
        elif R == 'star':
            model = Baseline(pool_style='hi_pool', n_classes=17, seq_len=240).cuda()
        else:
            model = Baseline(pool_style='hi_pool_fixed', n_classes=17, seq_len=240).cuda()
            model.pool.set_R(R=R)
            model.name += '-R={}'.format(R)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=50,
                                                    gamma=0.98)

        learner = Learner(model=model, dataset=(train_dataset, test_dataset, eval_dataset),
                          loss_function=tagging_loss, optimizer=optimizer, scheduler=scheduler,
                          options=options)
        print('Pool:{}'.format(model.name))
        if not options.eval_only:
            learner.learning(resume=False)
        result = write_metrics(checkpoint='f1_tag')
        results.append(result)
    pkl.dump(results, open('temps/data/ablation_results.pkl', 'wb'))
    
