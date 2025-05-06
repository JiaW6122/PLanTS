# !pip install einops
import seaborn as sns
from dataclasses import dataclass
import matplotlib.pyplot as plt

import datautils
from utils import init_dl_program
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
# data_names= [
#              'ArticularyWordRecognition',
#              'AtrialFibrillation',
#              'Cricket',
#              # 'CharacterTrajectories',
#              'BasicMotions',
#              'Epilepsy',
#              'EthanolConcentration',
#              'ERing',
#              # 'FaceDetection',
#              'FingerMovements',
#              'HandMovementDirection',
#              'Handwriting',
#              # 'EigenWorms',
#              'Heartbeat',
#              'DuckDuckGeese',
#              # 'InsectWingbeat',
#              'Libras',
#              'LSST',
#              'JapaneseVowels',
#              'NATOPS',
#              # 'PenDigits',
#              # 'PEMS-SF',
#              'MotorImagery',
#              # 'PhonemeSpectra',
#              'SelfRegulationSCP1',
#              'RacketSports',
#              'StandWalkJump',
#              'SelfRegulationSCP2',
#              'SpokenArabicDigits',
#              'UWaveGestureLibrary']
data_names = [ "BasicMotions"]
data_acc = defaultdict(float)  # save the acc of each dataset, keys are names of datasets

#########################################################
# train and test on each dataset
#########################################################
i=0
for dataset in data_names:
    i=i+1
    clear_output(wait=True)
    print(f"Running on {dataset}, num {i}")
    train_data, train_labels, test_data, test_labels = datautils.load_UEA(dataset)
    start_time = time.time()
    args = Args(
        static_repr_dims=128,
        dynamic_repr_dims=128,
        task_weights={
            'local_static_contrast': 0.25,
            'global_vatiant_contrast': 0.25,
            'dynamic_trend_pred': 0.5,
        },
        eval=False,
    )
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    model = HDST(
        input_dims=train_data.shape[-1],
        device=device,
        task_weights=args.task_weights,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims1=args.static_repr_dims,
        output_dims2=args.dynamic_repr_dims,
        max_train_length=args.max_train_length
    )
    
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        k=args.sample_size,
        w=args.window_size
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Model training time: {training_time:.2f} seconds")
    
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    fit_clf = eval_protocols.fit_svm
    clf = fit_clf(train_repr, train_labels)
    acc = clf.score(test_repr, test_labels)
    y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    print( 'acc:', acc, 'auprc:', auprc )
    data_acc[dataset] = acc
    with open('classification_accuracy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Accuracy'])  # optional header
        for dataset, acc in data_acc.items():
            writer.writerow([dataset, acc])

#########################################################
# show the final results
#########################################################
# print(data_acc)
average_acc = sum(data_acc.values()) / len(data_acc) if data_acc else 0.0
print(average_acc)
with open('classification_accuracy.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'Accuracy'])  # optional header
    for dataset, acc in data_acc.items():
        writer.writerow([dataset, acc])