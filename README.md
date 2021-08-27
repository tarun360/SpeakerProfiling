<!---# Commands

```bash
python train_timit.py --n_workers=0 --data_path='/notebooks/dataset/wav_data' --speaker_csv_path='/notebooks/SpeakerProfiling/Dataset/data_info_height_age.csv' --noise_dataset_path='/notebooks/noise_dataset'
```

```bash
python train_timit.py --n_workers=0 --data_path=/notebooks/SpeakerProfiling/TIMIT_Dataset/wav_data/ --speaker_csv_path=/notebooks/SpeakerProfiling/Dataset/data_info_height_age.csv
```

```bash
python test_timit.py --data_path=/notebooks/SpeakerProfiling/TIMIT_Dataset/wav_data/ --model_checkpoint=checkpoints/epoch=1-step=245-v3.ckpt
```
-->

# Speaker Profiling

This Repository contains the code for estimating the Age and Height of a speaker with their speech signal. This repository uses [s3prl](https://github.com/s3prl/s3prl) library to load various upstream models like wav2vec2, CPC, TERA etc. This repository uses TIMIT dataset. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages for preparing the dataset, training and testing the model.

```bash
pip install -r requirements.txt
```

## Usage

### Download the TIMIT dataset
```bash
wget https://data.deepai.org/timit.zip
unzip timit.zip -d 'path to timit data folder'
```

### Prepare the dataset for training and testing
```bash
python TIMIT/prepare_timit_data.py --path='path to timit data folder'
```

### Update Config and Logger
Update the config.py file to update the upstream model, batch_size, gpus, lr, etc and change the preferred logger in train_.py files

### Training
```bash
python train_timit.py --data_path='path to final data folder' --speaker_csv_path='path to this repo/SpeakerProfiling/Dataset/data_info_height_age.csv'
```

Example:
```bash
python train_timit.py --data_path=/notebooks/SpeakerProfiling/TIMIT_Dataset/wav_data/ --speaker_csv_path=/notebooks/SpeakerProfiling/Dataset/data_info_height_age.csv
```

### Testing
```bash
python test_timit.py --data_path='path to final data folder' --model_checkpoint='path to saved model checkpoint'
```

Example:
```bash
python test_timit.py --data_path=/notebooks/SpeakerProfiling/TIMIT_Dataset/wav_data/ --model_checkpoint=checkpoints/epoch=1-step=245-v3.ckpt
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference
- [1] S3prl: The self-supervised speech pre-training and representation learning toolkit. AT Liu, Y Shu-wen

