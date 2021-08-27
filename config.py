import os

class TIMITConfig(object):
    # path to the unzuipped TIMIT data folder
    data_path = '/notebooks/SpeakerProfiling/TIMIT_Dataset/wav_data/'

    # path to csv file containing age, heights of timit speakers
    speaker_csv_path = os.path.join(str(os.getcwd()), '/notebooks/SpeakerProfiling/Dataset/data_info_height_age.csv')

    # length of wav files for training and testing
    timit_wav_len = 3 * 16000
    # 16000 * 2

    batch_size = 32
    epochs = 200
    
    # loss = alpha * height_loss + beta * age_loss + gamma * gender_loss
    alpha = 1
    beta = 1
    gamma = 1

    # training type - AHG/H
    training_type = 'AHG'

    # data type - raw/spectral
    data_type = 'raw' 

    # model type
    ## AHG 
    # wav2vecTransformer
    
    ## H
    # wav2vecTransformer
    model_type = 'wav2vecTransformer'

    # number of layers in encoder (transformers)
    num_layers = 6

    # feature dimension of upstream model. For example, incase of wav2vec2, feature_dim = 768
    feature_dim = 768

    # No of GPUs for training and no of workers for datalaoders
    gpu = '-1'
    n_workers = 0

    # model checkpoint to continue from
    model_checkpoint = None
    
    # noise dataset for augmentation
    noise_dataset_path = '/home/shangeth/noise_dataset'

    # LR of optimizer
    lr = 1e-3

    run_name = data_type + '_' + training_type + '_' + model_type
