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

### Results

#### Multitask learning: height & age estimation and gender classification on TIMIT dataset using wav2vec2:

| Model                                   | lr     | Height RMSE |        | Height MAE |        | Age RMSE |        | Age MAE |        | Gender Accuracy | Epochs | Optimiser | batch size | Multitask    | train-augmentation | test-augmentation |
| --------------------------------------- | ------ | ----------- | ------ | ---------- | ------ | -------- | ------ | ------- | ------ | --------------- | ------ | --------- | ---------- | ------------ | ------------------ | ----------------- |
|                                         |        | Male        | Female | Male       | Female | Male     | Female | Male    | Female |                 |        |           |            |              |                    |                   |
| wav2vec2(frozen)+encoder-6L (multitask) | 0.1    | 7.82        | 11.7   | 6.18       | 10.22  | 8.26     | 9.25   | 5.43    | 6.29   | 66.66           | 25     | Adam      | 32         | A,H,G, 1,1,1 | PadCrop, Clipping  | PadCrop           |
| wav2vec2(frozen)+encoder-6L (multitask) | 0.01   | 7.87        | 11.57  | 6.24       | 10.09  | 8.06     | 9.11   | 5.76    | 6.48   | 66.66           | 25     | Adam      | 32         | A,H,G, 1,1,1 | PadCrop, Clipping  | PadCrop           |
| wav2vec2(frozen)+encoder-6L (multitask) | 0.001  | 7.72        | 11.95  | 6.07       | 10.46  | 8.28     | 9.26   | 5.42    | 6.28   | 66.66           | 25     | Adam      | 32         | A,H,G, 1,1,1 | PadCrop, Clipping  | PadCrop           |
| wav2vec2(frozen)+encoder-6L (multitask) | 0.0001 | 7.5         | 7.13   | 5.8        | 5.58   | 7.02     | 7.55   | 4.58    | 5.02   | 99.52           | 25     | Adam      | 32         | A,H,G, 1,1,1 | PadCrop, Clipping  | PadCrop           |

#### Multitask learning: height & age estimation and gender classification on TIMIT dataset using npc:

| Model                              | lr      | Height RMSE |       | Height MAE |       | Age RMSE |      | Age MAE |      | Gender Accuracy | Epochs | Optimiser | batch size | Multitask    | train-augmentation | test-augmentation |
| ---------------------------------- | ------- | ----------- | ----- | ---------- | ----- | -------- | ---- | ------- | ---- | --------------- | ------ | --------- | ---------- | ------------ | ------------------ | ----------------- |
| npc(frozen)+encoder-6L (multitask) | 0.001   | 7.89        | 11.53 | 6.25       | 10.05 | 8.26     | 9.25 | 5.43    | 6.29 | 66.66           | 30     | Adam      | 64         | A,H,G, 1,1,1 | PadCrop, Clipping  | PadCrop           |
| npc(frozen)+encoder-6L (multitask) | 0.0001  | 8.06        | 6.91  | 6.17       | 5.44  | 8.03     | 8.33 | 5.65    | 6.31 | 99.4            | 30     | Adam      | 64         | A,H,G, 1,1,1 | PadCrop, Clipping  | PadCrop           |
| npc(frozen)+encoder-6L (multitask) | 0.00001 | 7.61        | 6.66  | 5.66       | 5.24  | 7.99     | 7.72 | 5.41    | 5.48 | 98.8            | 30     | Adam      | 64         | A,H,G, 1,1,1 | PadCrop, Clipping  | PadCrop           |

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference
- [1] S3prl: The self-supervised speech pre-training and representation learning toolkit. AT Liu, Y Shu-wen

