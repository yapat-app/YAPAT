from MainClasses.DataHandler import DataHandler
import os

if __name__ == '__main__':
    dataset = 'DCASE2017'
    dh = DataHandler(dataset) # 240 nframes
    output_path = os.path.join(dh.root, 'features_400')
    # dh.get_feats(audio_path=os.path.join(dh.root, 'audio/training'), 
    #              type='training', outpath=output_path)
    # dh.get_feats(audio_path=os.path.join(dh.root, 'audio/testing'), 
    #              type='testing', outpath=output_path)
    # dh.get_feats(audio_path=os.path.join(dh.root, 'audio/evaluation'), 
    #              type='evaluation', outpath=output_path)

    dh.get_weak_labels_dcase(os.path.join(dh.root, 'groundtruth_weak_label_training_set.txt'), type='training', outpath=output_path)
    dh.get_strong_labels_dcase(os.path.join(dh.root, 'groundtruth_strong_label_testing_set.txt'), type='testing', outpath=output_path)
    dh.get_strong_labels_dcase(os.path.join(dh.root, 'groundtruth_strong_label_evaluation_set.txt'), type='evaluation', outpath=output_path)
