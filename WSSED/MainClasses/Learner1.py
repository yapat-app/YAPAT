import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from MainClasses.Metrics import Metrics
import pandas as pd
from MainClasses.Evaluator import Evaluator
from MainClasses.MILPooling import MILPooling
pd.set_option('precision', 8)


class Learner:
    def __init__(self, model, dataset, loss_function, optimizer, options, pool_fn=None, scheduler=None, fold=0):
        self.training_set = dataset[0]
        self.testing_set = dataset[1]
        self.evaluation_set = dataset[2]
        self.model = model
        self.options = options
        self.loss_fn = loss_function
        self.optimizer = optimizer
        if pool_fn is not None:
            self.pool = pool_fn
        self.k = fold
        self.scheduler = scheduler

    def learning(self, resume=False):
        train_loader = DataLoader(self.training_set, batch_size=self.options.batch_size,
                                  shuffle=True, num_workers=self.options.num_workers,
                                  drop_last=True)
        epoch = self.options.epoch
        max_f1_tag, max_f1_loc, min_er = 0, 0, 1e3
        val_f1_tag, val_f1_loc = [], []
        max_f1_tag_e, max_f1_loc_e, min_er_e = 0, 0, 0
        if resume:
            stat_dict = torch.load('/data0/gaolj/sed_data/dcase/results/{}/fold-{}-last_epoch.h5'.format(
                self.model.name, 0), map_location=torch.device('cpu'))
            self.model.load_state_dict(stat_dict)

        for e in range(epoch + 1):
            self.model.train()
            training_bar = tqdm(train_loader)
            training_bar.set_description('Fold-{}, training epoch {}/{}'.format(self.k, e, epoch))
            loss_data = 0
            loss_margin = 0
            for n_sample, (x_data, y_data, y_tagging, _) in enumerate(training_bar):
                y_tagging_hat, yi_ = self.model(x_data.float().cuda())
                # l_tagging = self.loss_fn(y_tagging_hat, y_tagging.float().cuda())
                # mid_max = torch.max(alpha[:, 1:-1], dim=-1)[0]
                # l_margin = torch.max(torch.zeros(17).cuda(), (alpha[:, 0] - mid_max)) \
                #            + torch.max(torch.zeros(17).cuda(), (alpha[:, -1] - mid_max))
                l_tagging = self.loss_fn(y_tagging_hat, y_tagging.float().cuda()).mean(0)
                self.optimizer.zero_grad()
                loss_data += l_tagging.mean().cpu().item()
                # loss_margin += l_margin.mean().cpu().item()
                l_tagging.mean().backward()
                self.optimizer.step()
                training_bar.set_postfix_str('loss:{:.4f}'.format(
                    loss_data / (n_sample + 1)
                ))

            self.scheduler.step()

            if e % 10 == 0:
                self.model.eval()
                loss, (f1_tag, precision, recall), f1_loc, er = self.evaluation(type='testing')
                print('val result ==> loss:{:.4f}, er:{:.3f}, r_loc:{:.2f}, p_loc:{:.2f}, f1_loc:{:.2f}, f1_tag:{:.2f}'.format(
                    loss, er, f1_loc["recall"], f1_loc["precision"], f1_loc["f_measure"], f1_tag
                ))

                if f1_tag >= max_f1_tag:
                    max_f1_tag = f1_tag
                    max_f1_tag_e = e
                    torch.save(self.model.state_dict(), '/data0/gaolj/sed_data/dcase/results/{}/fold-{}-checkpoint-f1_tag.h5'.format(
                        self.model.name, self.k))

                if f1_loc["f_measure"] >= max_f1_loc:
                    max_f1_loc = f1_loc["f_measure"]
                    max_f1_loc_e = e
                    torch.save(self.model.state_dict(), '/data0/gaolj/sed_data/dcase/results/{}/fold-{}-checkpoint-f1_loc.h5'.format(
                        self.model.name, self.k))

                if er <= min_er:
                    min_er = er
                    min_er_e = e
                    torch.save(self.model.state_dict(), '/data0/gaolj/sed_data/dcase/results/{}/fold-{}-checkpoint-er.h5'.format(
                        self.model.name, self.k))

                print('fold-{} maximum f1_tag is {:.2f} in {}'.format(self.k, max_f1_tag, max_f1_tag_e))
                print('fold-{} maximum f1_loc is {:.2f} in {}'.format(self.k, max_f1_loc, max_f1_loc_e))
                print('fold-{} minimum er is {:.2f} in {}'.format(self.k, min_er, min_er_e))

                val_f1_loc.append(f1_loc["f_measure"])
                val_f1_tag.append(f1_tag)
        torch.save(self.model.state_dict(), '/data0/gaolj/sed_data/dcase/results/{}/fold-{}-last_epoch.h5'.format(
            self.model.name, self.k))

    def evaluation(self, type, online=True):
        if not online:
            stat_dict = torch.load('/data0/gaolj/sed_data/dcase/results/' + self.model.name + '/fold-0-checkpoint-f1_tag.h5',
                                   map_location=torch.device('cpu'))
            self.model.load_state_dict(stat_dict)
        self.model.eval()
        if type == "testing":
            set = self.testing_set
            gt_path = "/data0/gaolj/sed_data/dcase/groundtruth_strong_label_testing_set.txt"
        elif type == "evaluation":
            set = self.evaluation_set
            gt_path = "/data0/gaolj/sed_data/dcase/groundtruth_strong_label_evaluation_set.txt"
        else:
            print("ERROR evaluation type!")
            return
        test_loader = DataLoader(set, batch_size=self.options.batch_size,
                                 shuffle=True, num_workers=self.options.num_workers)

        evaluator = Evaluator(self.options, self.loss_fn, self.model, test_loader)
        evaluator.forward()
        loss, tag_f1, precision, recall = evaluator.tag_evaluate()
        results = evaluator.loc_evaluate(strong_gt_path=gt_path)
        loc_f1 = results["overall"]["f_measure"]
        loc_er = results["overall"]["error_rate"]["error_rate"]
        return loss, (tag_f1, precision, recall), loc_f1, loc_er