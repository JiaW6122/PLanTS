# PLanTS 🌱
This repository contains the official implementation for the paper: **PLanTS: Periodicity-aware Latent-state Representation Learning for Multivariate Time Series**. **PLanTS** is a self-supervised learning framework for non-stationary multivariate time series.  


## Requirments
Install dependencies:
```bash
pip install -r requirements.txt
```


## Datasets
We evaluate PLanTS on multiple benchmark multivariate time series datasets:

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), a large publicly available electrocardiography dataset. Download the dataset and put it into `datasets/ptb-xl`. Then preprocess the dataset using `python datasets/preprocess_PTB-XL.py --group <group>`. `<group>` is the label group. Chosing from "diagnostic", "form" and "rhythm".

[UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), Human Activity Recognition database built from the recordings of 30 subjects performing activities of daily living (ADL). Download the dataset and put it into `datasets/HAR_data`. Then preprocess the dataset using `python datasets/preprocess_HAR.py`.

The access and preprocessing of 30 UEA datasets, 4 ETT datasets and Yahoo dataset can be found in [TS2Vec](https://github.com/zhihanyue/ts2vec).

##Usage
Train and test scripts for different tasks are provided:
- [train_test_classification.py] 
- [train_test_forecasting.py]   
- [train_test_anomaly.py] 
- [train_test_multi-label_classification.py] 
- [train_test_state_prediction.py]

Example command:

```bash
python train_test_classification.py --dataset_name=<dataset_name>
```
Replace `<dataset_name>` with your dataset name.

##Citation