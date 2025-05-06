# !pip install einops
import seaborn as sns
from dataclasses import dataclass
import matplotlib.pyplot as plt

import datautils
from utils import init_dl_program,FFT_for_Period,split_with_nan,centerize_vary_length_series
from hdst import HDST
import torch
import gc
import numpy as np
import os
import time
from IPython.display import clear_output
from tasks import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader

import csv
from tqdm import tqdm

#########################################################
# set hyper-parameters
#########################################################
@dataclass
class Args:
    task_weights: dict
    dataset: str = ""
    loader: str = ""
    gpu: int = 0
    static_repr_dims: int = 128
    dynamic_repr_dims: int = 128
    epochs: int = 200

    run_name: str = ""
    batch_size: int = 128
    lr: float = 0.001
    max_train_length = 800
    iters: int = None
    save_every = None
    seed: int = 1234
    max_threads = None
    eval: bool = True
    irregular = 0

    sample_size: int = 20
    window_size: int = 10

# args = Args(
#     static_repr_dims=128,
#     dynamic_repr_dims=128,
#     task_weights={
#         'local_static_contrast': 0.1,
#         'global_vatiant_contrast': 0,
#         'dynamic_trend_pred': 0.9,
#     },
#     eval=False,
# )
# device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

#########################################################
# get all the dataset names to run
#########################################################
# data_dir = "datasets/UEA"
# data_names = [name for name in os.listdir(data_dir)
#            if os.path.isdir(os.path.join(data_dir, name))]
# data_names= ['MotorImagery',
#              'PhonemeSpectra',
#              'DuckDuckGeese',
#              'SelfRegulationSCP1',
#              'RacketSports',
#              'StandWalkJump',
#              'SelfRegulationSCP2',
#              'UWaveGestureLibrary']
data_names= [
             'ArticularyWordRecognition',
             'AtrialFibrillation',
             'Cricket',
             'CharacterTrajectories',
             'BasicMotions',
             'Epilepsy',
             'EthanolConcentration',
             'ERing',
             'FaceDetection',
             'FingerMovements',
             'HandMovementDirection',
             'Handwriting',
             'EigenWorms',
             'Heartbeat',
             'DuckDuckGeese',
             'InsectWingbeat',
             'Libras',
             'LSST',
             'JapaneseVowels',
             'NATOPS',
             'PenDigits',
             'PEMS-SF',
             'MotorImagery',
             'PhonemeSpectra',
             'SelfRegulationSCP1',
             'RacketSports',
             'StandWalkJump',
             'SelfRegulationSCP2',
             'SpokenArabicDigits',
             'UWaveGestureLibrary']
# data_names = [ "BasicMotions"]
data_acc = defaultdict(float)  # save the acc of each dataset, keys are names of datasets








def check_period(train_data, top_k=5, max_train_length=800,batch_size=128):
    assert train_data.ndim == 3
    # Split data into windows, pad windows with nans to have equal lengths
    if max_train_length is not None:
        sections = train_data.shape[1] // max_train_length
        if sections >= 2:
            train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

    # What timesteps have no modalities present for at least one batch element
    temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
    if temporal_missing[0] or temporal_missing[-1]:
        train_data = centerize_vary_length_series(train_data)

    # Eliminate empty series        
    train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
    
    print(f"Training data shape: {train_data.shape}")
    train_dataset = (torch.from_numpy(train_data).to(torch.float))

    scale_list, scale_weight = FFT_for_Period(train_dataset, top_k)

    
    

    # # print(len(train_dataset))
    # train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True, drop_last=True)
    # print(train_loader)

    

    
    # for batch in train_loader:
    #     # Batch is a 1 element list
    #     x = batch[0]
    #     # print(x.shape)
    #     if max_train_length is not None and x.size(1) > max_train_length:
    #         window_offset = np.random.randint(x.size(1) - max_train_length + 1)
    #         x = x[:, window_offset : window_offset + max_train_length]

        
    #     scale_list, scale_weight = FFT_for_Period(x, top_k)
    print(scale_list)
    return(scale_list)


#########################################################
# train and test on each dataset
#########################################################
i=0
period_list={}
for dataset in data_names:
    i=i+1
    clear_output(wait=True)
    print(f"Running on {dataset}, num {i}")
    train_data, train_labels, test_data, test_labels = datautils.load_UEA(dataset)
    scale_list=check_period(train_data)
    period_list[dataset]=scale_list

#########################################################
# show the final results
#########################################################
# print(data_acc)
with open('period_list.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'Period'])  # optional header
    for dataset, per in period_list.items():
        writer.writerow([dataset, per])