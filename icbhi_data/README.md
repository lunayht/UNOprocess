# ICBHI'17 Respiratory Sound Analysis
This database consists of 5.5 hours of digital stethoscope recordings collected in Portugal and Greece. There are 920 annotated stehoscope audio samples in total, with lengths ranging from 10s to 90s, recorded using a variety of devices, acquisition modes, and patient demographics. 

## Data Preparation
0. Download the dataset from the official website: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge and unzip it.
1. In this repository, we follow the official 60/40 train-test split that is specified in `icbhi_train_test_split.csv`.
2. We have provided a preprocessing script `prep_data.py` that can slice the audio files into breathing cycles/ user-specified length, resample into a target sample rate, and save the processed audio into a new folder. You can run the script by executing the following command:
```bash
python prep_data.py
```
