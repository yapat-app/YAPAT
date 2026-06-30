import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from MainClasses.Metrics import Metrics
import pandas as pd
from MainClasses.Evaluator import Evaluator
from MainClasses.MILPooling import MILPooling
from scipy.io import savemat
import pudb
import os
pd.set_option('precision', 8)


class Learner:
    def __init__(self, model, dataset, loss_function, optimizer, options, scheduler=None):
        self.training_set = dataset[0]
        self.testing_set = dataset[1]
        self.evaluation_set = dataset[2]
        self.result_path =  options.result_path
        self.model = model
        self.options = options
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.root_path = self.options.config['data']['root']

    def learning(self, resume=False):
        train_loader = DataLoader(self.training_set, batch_size=self.options.batch_size,
                                  shuffle=True, num_workers=self.options.num_workers,
                                  drop_last=True)
        epoch = self.options.epoch
        max_f1_tag, max_f1_event = 0, 0
        max_f1_tag_e, max_f1_event_e = 0, 0
        if resume:
            print('Resume Mode=========>\n')
            stat_dict = torch.load(self.result_path + '/{}/checkpoint-f1_tag.h5'.format(
                self.model.name), map_location=torch.device('cpu'))
            self.model.load_state_dict(stat_dict)
            loss, (f1_tag, precision, recall), f1_loc, er = self.evaluation(type='testing')
            max_f1_tag, max_f1_loc, min_er = f1_tag, f1_loc['f_measure'], er
        loss_e = []
        f1_e_model, grad_e_alpha = [], []
        for e in range(epoch + 1):
            self.model.train()
            training_bar = tqdm(train_loader)
            training_bar.set_description('training epoch {}/{}'.format(e, epoch))
            loss_data = 0
            for n_sample, (x_data, y_tagging, audio_file) in enumerate(training_bar):
                tic_ = time.time()
                y_tagging_hat, yi_ = self.model(x_data.float().cuda())
                toc_ = time.time()
                print(toc_-tic_)
                l_tagging = self.loss_fn(y_tagging_hat, y_tagging.float().cuda()).mean()
                self.optimizer.zero_grad()
                loss_data += l_tagging.cpu().item()
                l_tagging.backward()
                self.optimizer.step()
                training_bar.set_postfix_str('loss:{:.4f}'.format(
                    loss_data / (n_sample + 1)
                ))

            loss_e.append(loss_data / (n_sample + 1))
            if not e > 200:
                self.scheduler.step()
            if e % 10 == 0:
                self.model.eval()
                loss, (f1_tag, precision, recall), f1_loc, er_loc, f1_event, er_event = self.evaluation(type='testing')
                print('val result ==> loss:{:.4f}, f1_tag:{:.2f}, f1_loc:{:.2f}, er_loc:{:.3f}, f1_event:{:.3f}, er_event:{:.3f}'.format(
                    loss, f1_tag, f1_loc, er_loc, f1_event, er_event
                ))

                if f1_tag >= max_f1_tag:
                    max_f1_tag = f1_tag
                    max_f1_tag_e = e
                    torch.save(self.model.state_dict(), self.result_path + '/{}/checkpoint-f1_tag.h5'.format(
                        self.model.name))
                
                if f1_event >= max_f1_event:
                    max_f1_event = f1_event
                    max_f1_event_e = e
                    torch.save(self.model.state_dict(), self.result_path + '/{}/checkpoint-f1_event.h5'.format(
                        self.model.name))

                print('maximum f1_tag is {:.2f} in {}'.format(max_f1_tag, max_f1_tag_e))
                print('maximum f1_event is {:.2f} in {}'.format(max_f1_event, max_f1_event_e))
                print("LR:{}".format(self.scheduler.get_last_lr()))

        torch.save(self.model.state_dict(), self.result_path + '/{}/last_epoch.h5'.format(
            self.model.name))
        # torch.save((loss_e, grad_e_alpha, grad_e_model), 'results/grad_zeros.h5')

    def evaluation(self, type, cp='f1_tag', online=True, avg=True):
        if not online:
            stat_dict = torch.load(os.path.join(self.result_path, self.model.name, 'checkpoint-' + cp + '.h5'),
                                   map_location=torch.device('cpu'))
            self.model.load_state_dict(stat_dict)
        self.model.eval()
        if type == "testing":
            set = self.testing_set
            if self.options.dataset == 'DCASE2017':
                gt_path = self.root_path + "/groundtruth_strong_label_testing_set.txt"
            elif self.options.dataset == 'DCASE2018':
                gt_path = self.root_path + "/metadata/test/test.csv"
            elif self.options.dataset == 'DCASE2019':
                gt_path = self.root_path + "/metadata/test/test_2019.csv"
            else:
                return

        elif type == "evaluation":
            set = self.evaluation_set
            if self.options.dataset == 'DCASE2017':
                gt_path = self.root_path + "/groundtruth_strong_label_evaluation_set.txt"
            elif self.options.dataset == 'DCASE2018':
                gt_path = self.root_path + "/metadata/eval/eval.csv"
            elif self.options.dataset == 'DCASE2019':
                gt_path = self.root_path + "/metadata/eval/public.tsv"
            else:
                return
        else:
            print("ERROR evaluation type!")
            return
        test_loader = DataLoader(set, batch_size=self.options.batch_size,
                                 shuffle=True, num_workers=self.options.num_workers)

        evaluator = Evaluator(self.options, test_loader, self.loss_fn, self.model)
        evaluator.forward()
        loss, tag_f1, precision, recall = evaluator.tag_evaluate(avg=avg)
        results = evaluator.loc_evaluate(strong_gt_path=gt_path)
        
        loc_f1 = results["segment"]["f_measure"]
        loc_er = results["segment"]["error_rate"]
        event_f1 =  results["event"]["f_measure"]
        event_er = results["event"]["error_rate"]
        if avg:
            return loss, (tag_f1, precision, recall), loc_f1, loc_er, event_f1, event_er
        else:
            return tag_f1, results["segment"]['class_wise']



    @staticmethod
    def get_grad_norm_w(model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    @staticmethod
    def get_grad_norm_alpha(model):
        total_norm_evets = []
        for p in model.parameters():
            if p.grad is not None:
                for k in range(17):
                    param_norm = p.grad[k].data.norm(2)
                    param_norm = param_norm.item() ** 2
                    param_norm = param_norm ** (1. / 2)
                    total_norm_evets.append(param_norm)
        return total_norm_evets