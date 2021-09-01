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

Multitask learning: height & age estimation and gender classification.

| Model                                    | lr     | Height RMSE |        | Height MAE |        | Age RMSE |        | Age MAE |        | Epochs | Optimiser | batch size | Multitask    |
| ---------------------------------------- | ------ | ----------- | ------ | ---------- | ------ | -------- | ------ | ------- | ------ | ------ | --------- | ---------- | ------------ |
|                                          |        | Male        | Female | Male       | Female | Male     | Female | Male    | Female |        |           |            |              |
| wav2vec2(frozen)+encoder-6L (multitastk) | 0.1    | 7.82        | 11.7   | 6.18       | 10.22  | 8.26     | 9.25   | 5.43    | 6.29   | 25     | Adam      | 32         | A,H,G, 1,1,1 |
| wav2vec2(frozen)+encoder-6L (multitastk) | 0.01   | 7.87        | 11.57  | 6.24       | 10.09  | 8.06     | 9.11   | 5.76    | 6.48   | 25     | Adam      | 32         | A,H,G, 1,1,1 |
| wav2vec2(frozen)+encoder-6L (multitastk) | 0.001  | 7.72        | 11.95  | 6.07       | 10.46  | 8.28     | 9.26   | 5.42    | 6.28   | 25     | Adam      | 32         | A,H,G, 1,1,1 |
| wav2vec2(frozen)+encoder-6L (multitastk) | 0.0001 | 7.5         | 7.13   | 5.8        | 5.58   | 7.02     | 7.55   | 4.58    | 5.02   | 25     | Adam      | 32         | A,H,G, 1,1,1 |

Only height estimation.

| Model                      | lr     | Height RMSE |       | Height MAE |       | Age RMSE |      | Age MAE |      | Epochs | Optimiser | batch size |
| -------------------------- | ------ | ----------- | ----- | ---------- | ----- | -------- | ---- | ------- | ---- | ------ | --------- | ---------- |
| wav2vec2(frozen)+encoder-6 | 0.1    | 8.96        | 9.72  | 7.31       | 8.19  | 10.65    | 11.3 | 7.05    | 7.01 | 50     | Adam      | 128        |
| wav2vec2(frozen)+encoder-6 | 0.01   | 7.8         | 11.74 | 6.16       | 10.26 | 8.27     | 9.25 | 5.43    | 6.29 | 50     | Adam      | 128        |
| wav2vec2(frozen)+encoder-6 | 0.001  | 7.8         | 11.75 | 6.16       | 10.26 | 8.26     | 9.25 | 5.43    | 6.29 | 50     | Adam      | 128        |
| wav2vec2(frozen)+encoder-6 | 0.0001 | 7.75        | 6.23  | 5.82       | 5.02  | 7.04     | 7.45 | 4.83    | 5.05 | 50     | Adam      | 128        |


## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference
- [1] S3prl: The self-supervised speech pre-training and representation learning toolkit. AT Liu, Y Shu-wen

