# ICBHI'17 Respiratory Sound Analysis
This database consists of 5.5 hours of digital stethoscope recordings collected in Portugal and Greece. There are 920 annotated stehoscope audio samples in total, with lengths ranging from 10s to 90s, recorded using a variety of devices, acquisition modes, and patient demographics. In this study, we will prepare the dataset for the following tasks:
- Non-Contrastive Self-Supervised Learning
- Respiratory Sound Classification (RSC)
- Respiratory Disease Classification (RDC)

## Data Preparation
0. Download the dataset from the official website: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge and unzip it.
1. In this repository, we follow the official 60/40 train-test split that is specified in `icbhi_train_test_split.csv`.
2. We have provided a preprocessing script `prep_data.py` that can slice the audio files into breathing cycles/ user-specified length, resample into a target sample rate, and save the processed audio into a new folder. You can run the script by executing the following command:
```bash
python prep_data.py
```
Or use `--help` to see the available options.

## Non-Contrastive Self-Supervised Pre-training
To prepare the training data for self-supervised learning, we run the following commands:
```bash 
python prep_data.py --target-sample-rate 16000                   \ # resample to 16kHz for RSC
                    --ori_data_path /PATH/TO/ICBHI_ORI_WAV       \
                    --output_path /OUTPUT/PATH/FOR_RESAMPLED_WAV 
```
```bash
python prep_data.py --target-sample-rate 4000                    \ # resample to 4kHz for RDC
                    --ori_data_path /PATH/TO/ICBHI_ORI_WAV       \
                    --output_path /OUTPUT/PATH/FOR_RESAMPLED_WAV 
```
Note that we **do not slice** the audio into breathing cycles or certain segment length here. Therefore the pre-training data are the resampled audio samples from the official training set. 

## Respiratory Sound Classification (RSC)
For RSC, we will slice the audio into **breathing cycles** and resample the audio into target sample rate (16kHz, or any other target sample rate).
```bash
python prep_data.py --target-sample-rate 16000                   \ # resample to 16kHz for RSC
                    --ori_data_path /PATH/TO/ICBHI_ORI_WAV       \
                    --output_path /OUTPUT/PATH/FOR_RESAMPLED_WAV \
                    --ori_txt_path /PATH/TO/ICBHI_ORI_TXT        \
                    --slice
```

## Respiratory Disease Classification (RDC)
For RDC, we will slice the audio into **certain segment length without overlapping** (8s in this study) and resample the audio into target sample rate (4kHz, or any other target sample rate).
```bash
python prep_data.py --target-sample-rate 4000                    \ # resample to 4kHz for RDC
                    --ori_data_path /PATH/TO/ICBHI_ORI_WAV       \
                    --output_path /OUTPUT/PATH/FOR_RESAMPLED_WAV \
                    --slice_sec 8
```