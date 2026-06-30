import sed_eval
import numpy as np
from sklearn.metrics import precision_score, recall_score
import torch
import json
from tqdm import tqdm
from MainClasses.loc_vad import activity_detection
import pickle as pkl
import os
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pudb
import random


class Evaluator:
    def __init__(self, args, data_loader, loss_fn=None, model=None):
        self.model = model
        self.test_loader = data_loader
        # self.model.eval()
        self.args = args
        self.loss_fn = loss_fn
        self.output_dict = {}
        self.config = self.args.config
        n_classes = len(self.config['data']['labels'])
        self.tag_th = [self.config["eval"]["tag_threshold"]] * n_classes

    def generate_metadata(self, output_dict):
        """Write output to submission file.
        Args:
          output_dict: {
              'audio_name': (audios_num),
              'clipwise_output': (audios_num, classes_num),
              'framewise_output': (audios_num, frames_num, classes_num)}
          sed_params_dict: {
              'audio_tagging_threshold': float between 0 and 1,
              'sed_high_threshold': : float between 0 and 1,
              'sed_low_threshold': : float between 0 and 1,
              'n_smooth': int, silence between the same sound event shorter than
                  this number will be filled with the sound event
              'n_salt': int, sound event shorter than this number will be removed}
        """
        (audios_num, frames_num, classes_num) = output_dict['frame_out'].shape
        frames_per_second = frames_num // self.config["data"]["clip_len"]
        labels = self.config["data"]["labels"]

        event_list = []

        def _float_to_list(x):
            if 'list' in str(type(x)):
                return x
            else:
                return [x] * classes_num

        sed_params_dict = {}

        sed_params_dict['audio_tagging_threshold'] = _float_to_list(self.config["eval"]["tag_threshold"])
        sed_params_dict['sed_high_threshold'] = _float_to_list(self.config["eval"]["loc_threshold_high"])
        sed_params_dict['sed_low_threshold'] = _float_to_list(self.config["eval"]['loc_threshold_low'])
        sed_params_dict['n_smooth'] = _float_to_list(self.config["eval"]["smooth"])
        sed_params_dict['n_salt'] = _float_to_list(self.config["eval"]["smooth"])

        for n in range(audios_num):
            for k in range(classes_num):
                if output_dict['clip_out'][n, k] \
                        > sed_params_dict['audio_tagging_threshold'][k]:

                    bgn_fin_pairs = activity_detection(
                        x=output_dict['frame_out'][n, :, k],
                        thres=sed_params_dict['sed_high_threshold'][k],
                        low_thres=sed_params_dict['sed_low_threshold'][k],
                        n_smooth=sed_params_dict['n_smooth'][k],
                        n_salt=sed_params_dict['n_salt'][k])

                    for pair in bgn_fin_pairs:
                        if pair[0] >= pair[1]:
                            continue
                        event = {
                            'filename': output_dict['audio_name'][n],
                            'onset': pair[0] / float(frames_per_second),
                            'offset': pair[1] / float(frames_per_second),
                            'event_label': labels[k]}
                        event_list.append(event)
        return event_list

    def forward(self):
        self.model.eval()

        test_bar = tqdm(self.test_loader)
        output_dict = {'audio_name': [],
                       'clip_out': [],
                       'frame_out': [],
                       'target': [],
                       'strong_target': []}
        with torch.no_grad():
            # loss_data = 0
            for n_sample, (x_data, y_data, audio_file) in enumerate(test_bar):
                import time
                tic = time.time()
                y_tagging_hat, yi = self.model(x_data.float().cuda())
                toc = time.time()
                print(toc-tic)
                # y_tagging_hat = yi.mean(1)
                y_tagging = (y_data.cpu().numpy().sum(1) > 1).astype(float)
                output_dict['audio_name'] += list(audio_file)
                output_dict['clip_out'] += list(y_tagging_hat.cpu().numpy())
                output_dict['frame_out'] += list(yi.cpu().numpy())
                output_dict['target'] += list(y_tagging)
                output_dict['strong_target'] += list(y_data.cpu().numpy())
        # print('alpha:', alpha.argmax(-1))
        output_dict['clip_out'] = np.array(output_dict['clip_out'])
        output_dict['frame_out'] = np.array(output_dict['frame_out'])
        output_dict['target'] = np.array(output_dict['target'])
        output_dict['strong_target'] = np.array(output_dict['strong_target'])
        self.output_dict = output_dict

        # Framewise predictions to eventwise predictions
        predict_meta = self.generate_metadata(output_dict)
        if not os.path.exists(os.path.join(self.args.result_path, self.model.name)):
            os.mkdir(os.path.join(self.args.result_path, self.model.name))
        self.write_metadata(predict_meta, os.path.join(self.args.result_path, self.model.name, 'predict_meata.csv'))
        return output_dict

    def write_metadata(self, event_list, meta_path):
        """Write prediction event list to submission file for later evaluation.
        Args:
          event_list: list of events
          submission_path: str
        """

        f = open(meta_path, 'w')

        for event in event_list:
            f.write('{}\t{}\t{}\t{}\n'.format(
                event['filename'], event['onset'], event['offset'], event['event_label']))

    def tag_evaluate(self, avg=True):
        """Calculate clip level precision, recall, F1."""
        y_true = self.output_dict['target']
        y_pred = self.output_dict['clip_out']
        classes_num = y_true.shape[-1]
        binarized_output = np.zeros_like(y_pred)
        loss = self.loss_fn(torch.from_numpy(y_pred).float(), torch.from_numpy(y_true).float()).mean().numpy()
        for k in range(classes_num):
            binarized_output[:, k] = (np.sign(y_pred[:, k] - self.tag_th[k]) + 1) // 2
        if avg:
            precision = precision_score(y_true.flatten(), binarized_output.flatten())
            recall = recall_score(y_true.flatten(), binarized_output.flatten())
            f1 = 2 * precision * recall / (precision + recall)
        else:
            precision, recall, f1 = [], [], []
            for k in range(classes_num):
                p_ = precision_score(y_true[:, k], binarized_output[:, k])
                r_ = recall_score(y_true[:, k], binarized_output[:, k])
                f1_ = 2 * p_ * r_ / (p_ + r_ + 1e-6)
                precision.append(p_)
                recall.append(r_)
                f1.append(f1_)
        return loss, f1, precision, recall

    def loc_evaluate(self, strong_gt_path, avg=True):
        """Evaluate metrics with official SED toolbox.

        Args:
          reference_csv_path: str
          prediction_csv_path: str
        """
        reference_event_list = sed_eval.io.load_event_list(strong_gt_path,
                                                           delimiter='\t', csv_header=True,
                                                           fields=['filename', 'onset', 'offset', 'event_label'])

        estimated_event_list = sed_eval.io.load_event_list(os.path.join(self.args.result_path, self.model.name, 'predict_meata.csv'),
                                                           delimiter='\t', csv_header=False,
                                                           fields=['filename', 'onset', 'offset', 'event_label'])

        evaluated_event_labels = reference_event_list.unique_event_labels
        files = {}
        results = {
            'segment': {'f_measure': [], 'error_rate': []}, 
            'event': {'f_measure': [], 'error_rate': []}
            }
        for event in reference_event_list:
            files[event['filename']] = event['filename']

        evaluated_files = sorted(list(files.keys()))

        segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
            event_label_list=evaluated_event_labels,
            time_resolution=1.0
        )
        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=evaluated_event_labels,
            t_collar=0.200,
            percentage_of_length=0.2
        )
        for file in evaluated_files:
            reference_event_list_for_current_file = []
            for event in reference_event_list:
                if event['event_label'] is None:
                    event['onset'], event['offset'] = 0., 0.
                    event['event_label'] = random.sample(self.config['data']['labels'], 1)[0]
                if event['filename'] == file:
                    reference_event_list_for_current_file.append(event)
                    estimated_event_list_for_current_file = []
            for event in estimated_event_list:
                if len(event['filename'].split('/')) != len(file.split('/')):
                    event['filename'] = event['filename'].split('/')[-1]
                if event['filename'] == file or event['filename'][1:] == file:  # DCASE data has 'Y' tag in audio file names.
                    estimated_event_list_for_current_file.append(event)

            segment_based_metrics.evaluate(
                reference_event_list=reference_event_list_for_current_file,
                estimated_event_list=estimated_event_list_for_current_file
            )
            event_based_metrics.evaluate(
                reference_event_list=reference_event_list_for_current_file,
                estimated_event_list=estimated_event_list_for_current_file
            )
        segment_results = segment_based_metrics.results()
        event_results = event_based_metrics.results()
        results['segment']['f_measure'] = segment_results['overall']['f_measure']['f_measure'] # Micro Average
        results['segment']['error_rate'] = segment_results['overall']['error_rate']['error_rate'] # Micro Average
        results['event']['f_measure'] = event_results['class_wise_average']['f_measure']['f_measure'] # Macro Average
        results['event']['error_rate'] = event_results['class_wise_average']['error_rate']['error_rate'] # Macro Average
        results['segment']['class_wise'] = segment_results['class_wise']
        return results

    def loc_plot(self, model_list):
        def extend_arr(arr, scale):
            new_arr = np.zeros((arr.shape[0] * scale, arr.shape[1]))
            for i in range(len(arr)):
                new_arr[i * scale: (i + 1) * scale, :] = arr[i, :]
            return new_arr
        info_loc = {'files_list':[],
                      'strong_label':[],
                      'frame_outs':[],
                      'target': []}
        for model in model_list:
            stat_dict = torch.load(self.args.result_path + '/{}/checkpoint-f1_tag.h5'.format(model.name),
                                   map_location=torch.device('cpu'))
            model.load_state_dict(stat_dict)
            self.model = model
            out_ = self.forward()
            info_loc['files_list'] = out_['audio_name']
            info_loc['strong_label'] = out_['strong_target']
            info_loc['frame_outs'].append(out_['frame_out'])
            info_loc['target'] = out_['target']

        n_subplots = 4
        event_labels = self.config['data']['labels']
        sed_params_dict = {"tag_threshold": 0.5, "loc_threshold_high": 0.3,
                           "loc_threshold_low": 0.1, "n_smooth": 10, "n_salt": 10}
        # i = random.sample(list(range(len(info_loc['files_list']))), 1)[0]

        for i in range(len(info_loc['files_list'])):
            # file_name = info_loc['files_list'][i]
            file_name = 'YElJFYwRtrH4_30.000_40.000.wav'
            i = info_loc['files_list'].index(file_name)
            strong_label = info_loc['strong_label'][i]
            activated_events = np.where(info_loc['target'][i] == 1)[0]
            # if 10 not in activated_events:
            #     continue
            if len(activated_events) == 1:
                strong_label = np.concatenate((np.zeros((3, 240)), strong_label[:, activated_events[0]].reshape(1, 240),
                                np.zeros((3, 240)), np.zeros((1, 240)),
                                np.zeros((3, 240))))
            elif len(activated_events) == 0:
                continue
            else:
                strong_label = np.concatenate((np.zeros((3, 240)), strong_label[:, activated_events[0]].reshape(1, 240),
                                np.zeros((3, 240)), strong_label[:, activated_events[1]].reshape(1, 240),
                                np.zeros((3, 240))))

            fig, axs = plt.subplots(n_subplots, 1, figsize=(14, 6), dpi=200, sharex=True)
            axs[0].set_title(file_name, fontdict={'fontsize': 16, 'family': 'Times New Roman'})
            rm = axs[0].imshow(extend_arr(strong_label, 8), cmap='Greys', aspect='auto')
            axs[0].set_yticks(np.arange(4, 10 * 8 + 4, 8 * 2))
            # axs[0].set_yticklabels(np.arange(0, 20, 2), family='Times New Roman', size=14)
            # axs[0].set_yticklabels(["Train horn", "Air horn", "Car alarm", "Reversing beeps", "Bicycle", "Skateboard",
            #                       "Ambulance (siren)", "Fire engine", "Civil defense siren", "Police car",
            #                       "Screaming", "Car", "Car passing by", "Bus", "Truck", "Motorcycle", "Train"],
            #                      family='Times New Roman')
            # axs[0].set_yticklabels(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
            #                       "C11", "C12", "C13", "C14", "C15", "C16", "C17"], family='Times New Roman')

            axs[0].set_ylabel('Ground Truth', fontdict={'fontsize': 16, 'family': 'Times New Roman'})
            axs[0].set_yticklabels(['...', event_labels[activated_events[0]], '...', '...' if len(activated_events) == 1 else event_labels[activated_events[1]], '...'], rotation=35, fontdict={'fontsize': 12, 'family': 'Times New Roman'})
            titles = ['Reference', 'HiPool', 'MaxPool', 'AvgPool']
            for n in range(1, 4):
                axs[n].set_xticks(np.arange(0, 240, 24))
                axs[n].set_xticklabels(np.arange(10.0), fontdict={'fontsize': 14, 'family': 'Times New Roman'})
                smoothed_outs = np.zeros((240, 17))
                for k in range(17):
                    frame_out = info_loc['frame_outs'][n-1][i, :, k]
                    bgn_fin_pairs = activity_detection(
                        x=frame_out,
                        thres=sed_params_dict['loc_threshold_high'],
                        low_thres=sed_params_dict['loc_threshold_low'],
                        n_smooth=sed_params_dict['n_smooth'],
                        n_salt=sed_params_dict['n_salt'])
                    for pair in bgn_fin_pairs:
                        smoothed_outs[pair[0]:pair[1], k] = info_loc['frame_outs'][n-1][i, pair[0]:pair[1], k]
                preds = np.concatenate((np.zeros((3, 240)), smoothed_outs[:, activated_events[0]].reshape(1, 240),
                                                       np.zeros((3, 240)), np.zeros((1, 240)) if len(activated_events)==1 else smoothed_outs[:, activated_events[1]].reshape(1, 240),
                                                       np.zeros((3, 240))))
                # plt.subplot(n_subplots, 1, pool_style.index(pool) + 2)
                axs[n].imshow(extend_arr(preds, 8), cmap='Greys', aspect='auto')
                axs[n].set_yticks(np.arange(4, 10 * 8 + 4, 8 * 2))
                axs[n].set_yticklabels(['...', event_labels[activated_events[0]], '...', '...' if len(activated_events) == 1 else event_labels[activated_events[1]], '...'], rotation=35, fontdict={'fontsize': 12, 'family': 'Times New Roman'})
                axs[n].set_ylabel(titles[n], fontdict={'fontsize': 16, 'family': 'Times New Roman'})
            cb = fig.colorbar(rm, cmap='Greys', ax=axs)
            cb.ax.tick_params(size=14)
            fig.savefig(os.path.join(self.args.result_path, 'temp', file_name.split('.wav')[0] + '.png'), bbox_inches='tight')
            plt.close(fig)
            # fig.show()
            # print()



def scale_imshow(data, scale):
    new_data = np.zeros((data.shape[0], data.shape[1] * scale))
    for i in range(len(data[0])):
        new_data[:, i*scale:(i+1)*scale] = np.repeat(data[:, i].reshape(-1, 1), 3, 1)
    return new_data