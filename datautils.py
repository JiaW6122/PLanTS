import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 

from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import s3fs
import io

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset, normalize=True, s3_bucket=None, s3_prefix=None):
    if s3_bucket is not None:
        fs = s3fs.S3FileSystem()
        if s3_bucket.startswith('s3://'):
            s3_bucket = s3_bucket[5:]  # Remove 's3://'
 
        train_path = f'{s3_bucket}/{s3_prefix}/UEA/{dataset}/{dataset}_TRAIN.arff'
        test_path = f'{s3_bucket}/{s3_prefix}/UEA/{dataset}/{dataset}_TEST.arff'
 
        with fs.open(train_path, 'rb') as f:
            train_data = loadarff(io.TextIOWrapper(f, encoding='utf-8'))[0]
        with fs.open(test_path, 'rb') as f:
            test_data = loadarff(io.TextIOWrapper(f, encoding='utf-8'))[0]
    else:
        # Local mode
        train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
        test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
 
    # print(train_data.shape)
    # print(test_data.shape)
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
 
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
 
    if normalize:
        scaler = StandardScaler()
        scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
        train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
        test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
 
    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
 
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens


def load_forecast_csv(name, univar=False, raw=False, timeenc=0):
    data = pd.read_csv(f'datasets/{name}.csv')
    tmp_stamp = data[['date']]
    cols_data = data.columns[1:]
    data = data[cols_data]


    tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
    if timeenc == 0:
        tmp_stamp['month'] = tmp_stamp.date.apply(lambda row: row.month, 1)
        tmp_stamp['day'] = tmp_stamp.date.apply(lambda row: row.day, 1)
        tmp_stamp['weekday'] = tmp_stamp.date.apply(lambda row: row.weekday(), 1)
        tmp_stamp['hour'] = tmp_stamp.date.apply(lambda row: row.hour, 1)
        tmp_stamp['minute'] = tmp_stamp.date.apply(lambda row: row.minute, 1)
        tmp_stamp['minute'] = tmp_stamp.minute.map(lambda x: x // 15)
        data_stamp = tmp_stamp.drop(columns=['date']).values
    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(tmp_stamp['date'].values), freq='t')
        data_stamp = data_stamp.transpose(1, 0)

    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    if not raw:
        scaler = StandardScaler().fit(data[train_slice])
        data = scaler.transform(data)
    else:
        scaler = None

    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
    
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, data_stamp


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']

# def load_UEA(dataset, normalize=True, s3_bucket=None, s3_prefix=None):
#     if s3_bucket is not None:
#         fs = s3fs.S3FileSystem()
#         if s3_bucket.startswith('s3://'):
#             s3_bucket = s3_bucket[5:]  # Remove 's3://'
 
#         train_path = f'{s3_bucket}/{s3_prefix}/UEA/{dataset}/{dataset}_TRAIN.arff'
#         test_path = f'{s3_bucket}/{s3_prefix}/UEA/{dataset}/{dataset}_TEST.arff'
        
def load_ptb_xl(name,s3_bucket=None, s3_prefix=None):
    if s3_bucket is not None:
        fs = s3fs.S3FileSystem()
        if s3_bucket.startswith('s3://'):
            s3_bucket = s3_bucket[5:]  # Remove 's3://'
 
        x_train_path = f'{s3_bucket}/{s3_prefix}/{name}/x_train.pkl'
        x_test_path = f'{s3_bucket}/{s3_prefix}/{name}/x_test.pkl'
        x_val_path = f'{s3_bucket}/{s3_prefix}/{name}/x_val.pkl'
        y_train_path = f'{s3_bucket}/{s3_prefix}/{name}/y_train.pkl'
        y_test_path = f'{s3_bucket}/{s3_prefix}/{name}/y_test.pkl'
        y_val_path = f'{s3_bucket}/{s3_prefix}/{name}/y_val.pkl'
        with fs.open(x_train_path, 'rb') as f:
            x_train = pickle.load(f)
        with fs.open(x_test_path, 'rb') as f:
            x_test = pickle.load(f)
        with fs.open(x_val_path, 'rb') as f:
            x_val = pickle.load(f)
        with fs.open(y_train_path, 'rb') as f:
            y_train = pickle.load(f)
        with fs.open(y_test_path, 'rb') as f:
            y_test = pickle.load(f)
        with fs.open(y_val_path, 'rb') as f:
            y_val = pickle.load(f)
        

    else:
        x_train = pkl_load(f'datasets/ptb-xl/{name}/x_train.pkl')
        x_test = pkl_load(f'datasets/ptb-xl/{name}/x_test.pkl')
        x_val = pkl_load(f'datasets/ptb-xl/{name}/x_val.pkl')
        y_train = pkl_load(f'datasets/ptb-xl/{name}/y_train.pkl')
        y_test = pkl_load(f'datasets/ptb-xl/{name}/y_test.pkl')
        y_val = pkl_load(f'datasets/ptb-xl/{name}/y_val.pkl')
    return x_train, y_train, \
           x_test,  y_test,  \
           x_val,  y_val


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data

def load_HAR(name):
    x_train=pkl_load(f'datasets/{name}/x_train.pkl')
    state_train=pkl_load(f'datasets/{name}/state_train.pkl')
    x_test=pkl_load(f'datasets/{name}/x_test.pkl')
    state_test=pkl_load(f'datasets/{name}/state_test.pkl')
    return x_train, state_train, x_test, state_test


def visiualize_HAR(name,subject_id=1):
    with open('datasets/HAR_data/x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    with open('datasets/HAR_data/state_train.pkl', 'rb') as f:
        state_train = pickle.load(f)
    activity = pd.read_csv('datasets/HAR_data/activity_labels.txt',
                           header=None,
                           sep=' ',
                           names=['index', 'feature'])
    features = pd.read_csv('datasets/HAR_data/features.txt',
                           header=None,
                           sep=' ',
                           names=['index', 'feature'])
    x = x_train[subject_id]          # shape: (561, T)
    y = state_train[subject_id] 
    activity_labels = list(activity['feature'])

    # Assign fixed colors for activities (consistent coloring)
    activity_colors = {
        i: f'C{i}' for i in range(len(activity_labels))
    }
    
    T=x_train.shape[2]
    time=np.arange(T)
    plt.figure(figsize=(15, 6))
    for i in range(15):
        plt.plot(time, x[i],label=features['feature'][i])
    # Add background color blocks for activity labels
    current_label = y[0]
    start = 0
    for t in range(1, T):
        if y[t] != current_label:
            plt.axvspan(start, t, color=activity_colors[current_label], alpha=0.15)
            start = t
            current_label = y[t]
    plt.axvspan(start, T, color=activity_colors[current_label], alpha=0.15)
    
    # Add a custom legend for activity labels
    legend_patches = [
        Patch(facecolor=activity_colors[i], edgecolor='none', alpha=0.3, label=activity_labels[i])
        for i in range(len(activity_labels))
    ]
    legend_patches = [
        Patch(facecolor=activity_colors[i], edgecolor='none', alpha=0.3, label=activity_labels[i])
        for i in range(len(activity_labels))
    ]
    activity_legend = plt.legend(handles=legend_patches, title="Activity", loc='lower left', fontsize=9)
    plt.gca().add_artist(activity_legend)
    plt.title(f"Subject {subject_id} - Overlaid Features with Activity Background")
    plt.xlabel("Time steps")
    plt.ylabel("Feature values")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()



    
        