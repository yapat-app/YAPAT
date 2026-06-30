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
import pudb
pd.set_option('precision', 4)

DATASET = 'DCASE2017'   
def args_setup():
    parser = ArgumentParser()
    # NOTE: parameters below are not supposed to change.
    parser.add_argument("-dt", "--dataset", type=str, default=DATASET)
    parser.add_argument('-o', "--result_path", type=str, default='results/' + DATASET)
    parser.add_argument('-c', "--config", default=json.load(open('./MainClasses/config.json', 'r'))[DATASET])

    # Parameters below are changeable.
    parser.add_argument('-w', '--num_workers', type=int, default=16)
    parser.add_argument('-d', "--feature_dim", type=int, default=64)
    parser.add_argument('-b', "--batch_size", type=int, default=256)
    parser.add_argument('-e', "--epoch", type=int, default=100)
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.0003)
    parser.add_argument('-k', "--num_events", type=int, default=17)
    parser.add_argument('-gpu', "--gpu_device", type=str, default='0')
    parser.add_argument('--eval_only', action='store_true', default=False)
    """
    TODO: Change the pool style below in [max_pool, avg_pool, attention_pool, linear_pool, exp_pool, auto_pool, power_pool, hi_pool, hi_pool_plus]
    """
    parser.add_argument('-p', "--pooling", type=str, default='hi_pool_fixed')
    return parser


def write_metrics(checkpoint='f1_tag', dataset=DATASET):
    # evaluation, checkpoint = F1 on tagging
    print('============Evaluation==============')
    eval_data = 'evaluation'
    loss, (tag_f1, precision, recall), loc_f1, loc_er, event_f1, event_er = learner.evaluation(eval_data, cp=checkpoint, online=False)
    result_json = {
        'pool': model.name,
        'tagging': {
            'F1': tag_f1,
            'P': precision,
            'R': recall
        },
        'localization': {
            'F1': loc_f1,
            'ER': loc_er
        },
        'event-based':{
            'F1': event_f1,
            'ER': event_er
        }
    }
    with open(os.path.join('results/', dataset, model.name, eval_data + '_results.json'), 'w') as f:
        json.dump(result_json, f)
    print('Pool:{}'.format(result_json['pool']))
    print(result_json)

    # testing, checkpoint = F1 on tagging
    print('=============Testing=============')
    eval_data = 'testing'
    loss, (tag_f1, precision, recall), loc_f1, loc_er, event_f1, event_er = learner.evaluation(eval_data, cp=checkpoint, online=False)
    result_json = {
        'pool': model.name,
        'tagging': {
            'F1': tag_f1,
            'P': precision,
            'R': recall
        },
        'localization': {
            'F1': loc_f1,
            'ER': loc_er
        },
        'event-based':{
            'F1': event_f1,
            'ER': event_er
        }
    }
    with open(os.path.join('results/', dataset, model.name, eval_data + '_results.json'), 'w') as f:
        json.dump(result_json, f)
    print('Pool:{}'.format(result_json['pool']))
    print(result_json)


if __name__ == '__main__':
    args = args_setup()
    options = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_device
    batch_size = options.batch_size
    epoch = options.epoch
    lr = options.learning_rate

    pool_style = options.pooling
    tagging_loss = nn.BCELoss()

    train_dataset = DCASE2017(options, type_='training')
    test_dataset = DCASE2017(options, type_='testing')
    eval_dataset = DCASE2017(options, type_='evaluation')
    # for R in [2, 3, 4, 5, 6, 8, 10, 12, 15, 16]:
    for R in [20, 24, 30, 40, 48, 60, 80, 120]:
        model = Baseline(pool_style=pool_style, n_classes=17, seq_len=240).cuda()
        model.pool.set_R(R=R)
        model.name += '-R={}'.format(R)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=20,
                                                    gamma=0.90)

        learner = Learner(model=model, dataset=(train_dataset, test_dataset, eval_dataset),
                        loss_function=tagging_loss, optimizer=optimizer, scheduler=scheduler,
                        options=options)
        print('Pool:{}'.format(model.name))
        if not options.eval_only:
            learner.learning(resume=False)
        write_metrics(checkpoint='f1_tag')
