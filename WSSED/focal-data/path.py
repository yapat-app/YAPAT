BASE_PATH='/Users/iliratroshani/PycharmProjects/CRNN4SED/'

# MODEL
#MODEL_NAME = 'net_light_model'
#MODEL_NAME = 'Resnet_model'
#MODEL_NAME = 'vanilla_resnet_model'
MODEL_NAME = 'BirdnetEmbeddings_model'
#
#CHECKPOINT_NAME = str(12002)+'_1ep_fnjv_no_mixup_' + MODEL_NAME
CHECKPOINT_NAME = str(12002)+'_1ep_fnjv_no_mixup_batch_8' + MODEL_NAME
#CHECKPOINT_NAME = str(11669) +'_100ep_60sinput_without_da_with_masking_'+MODEL_NAME
#CHECKPOINT_NAME = str(11669) +'_100ep_60sinput_without_da_with_masking'+MODEL_NAME
                         #11669_100ep_60sinput_without_da_with_maskingnet_light_model
#CHECKPOINT_NAME = str(11788) +'_100ep_FNJV_var_sinput_without_da_with_masking_'+MODEL_NAME
#CHECKPOINT_NAME = str(11569) + 'train_on_60_test_on_3_' + MODEL_NAME
CHECKPOINTS_DIR = BASE_PATH + 'checkpoints/'

MODEL_PATH = CHECKPOINTS_DIR + CHECKPOINT_NAME+'.pt'
#MODEL_PATH = '/Users/iliratroshani/PycharmProjects/CRNN4SED/checkpoints/final2/11569_100ep_without_da_without_masking_net_light_model.pt'
#MODEL_PATH = '/Users/iliratroshani/PycharmProjects/CRNN4SED/checkpoints/11669_100ep_60sinput_without_da_with_maskingnet_light_model.pt'
#MODEL_PATH = '/Users/iliratroshani/PycharmProjects/CRNN4SED/checkpoints/final3/11669_100ep_60sinput_without_da_with_maskingnet_light_model.pt'
#MODEL_PATH = '/Users/iliratroshani/PycharmProjects/CRNN4SED/checkpoints/final3/11669_100ep_60sinput_without_da_net_light_model.pt'
#MODEL_PATH = '/Users/iliratroshani/PycharmProjects/CRNN4SED/checkpoints/11669_100ep_60sinput_without_da_with_maskingnet_light_model.pt'

METRICS_PATH = CHECKPOINTS_DIR + CHECKPOINT_NAME+'_metrics_thresh.csv'
TEST_RESULTS_PATH = CHECKPOINTS_DIR + CHECKPOINT_NAME+'_test_results_thresh.csv'
EVAL_PATH = CHECKPOINTS_DIR + CHECKPOINT_NAME+'_eval_thresh.csv'

# DATA
AUDIO_DIR = BASE_PATH + 'raw_data/'
#AUDIO_DIR = BASE_PATH + 'raw_data/FNJV/'
#AUDIO_DIR = BASE_PATH + 'raw_data/augmented/'
#AUDIO_DIR = BASE_PATH + 'raw_data/10_splits/'
#AUDIO_DIR = BASE_PATH + 'raw_data/3_splits/'

ANNOTATIONS_FILE = AUDIO_DIR + 'weak_labels.csv'
STRONG_LABELS_PATH = AUDIO_DIR + 'strong_labels'

class ModelPathManager():
    def __init__(self, model_name, seed, epochs, da):
        self.model_name = model_name
        self.seed = seed
        self.epochs = epochs
        self.da = da