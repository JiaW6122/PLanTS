!pip install einops
!pip install bottleneck
!pip install tslearn
%load_ext autoreload
%autoreload 2

import seaborn as sns
# from dataclasses import dataclass
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

import datautils
from utils import init_dl_program
from hdst import HDST
import torch
import gc
import numpy as np

SEPSIS_DATA_PATH = 'datasets/sepsis'
N_TEST_PATIENTS = 10084

def train_trep_sepsis(task_weights):

    ## 
    args = Args(
    dataset= "",
    loader = "",
    gpu= 0,
    static_repr_dims = 128,
    dynamic_repr_dims = 128,
    epochs: int = None,

    run_name = "",
    batch_size = 128,
    lr= 0.001,
    max_train_length = 800,
    iters = None,
    save_every = None,
    seed = 1234,
    max_threads = None,
    irregular = 0,

    sample_size = 20,
    # window_size: int = 10
    window_size = "Auto",
    # window_size: list = field(default_factory=lambda: [20, 50, 100, 200, 400, 800])
    distance = "mcc",
    top_k =3, # Use the top k prominent periodicity as time scale. If use sepcified window size, you can set top_k as random number.
    tmp_emb_type = "original", # Define the embedding type of time stamps. 
        #'temporal_fixed': using FixedEmbedding with explicit timestamps inputs. Like ETT
        #'temporal_learn': using nn.Embedding with explicit timestamps inputs.
        #'positional': using PositionalEmbedding with synthetic timestamps inputs. Like UEA
        #'original' : using original input to extract dynamic information. 
    freq = 'h', # 'h' for 'ETTh1', 'ETTh2', 't' for 'ETTm1' and 'ETTm2'
    task_weights=task_weights,
    eval=False,
    )

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)


    # Load data
    print(f"Loading anomaly detection data...")
    train_data = np.load(f'{SEPSIS_DATA_PATH}/processed_data/train_data.npy')
    test_data = np.load(f'{SEPSIS_DATA_PATH}/processed_data/test_data.npy')

    # Train and save model
    start_time = time.time()
    model = HDST(
        input_dims=train_data.shape[-1],
        tmp_embed_type=args.tmp_emb_type,
        freq=args.freq,
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
        w=args.window_size,
        distance=args.distance,
        top_k=args.top_k,
        n_channels=n_channels
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Model training time: {training_time:.2f} seconds")
    print("Training of embedding model done.")
    
    return model


def get_patient_windows(df, window_size):
    windows = []
    for i in range(window_size, len(df) + 1):
        windows.append(df.iloc[i - window_size:i])
    return windows


def get_window_xy(window_df):
    return window_df.drop(['SepsisLabel', 'ID'], axis=1).values, window_df['SepsisLabel'].max()


def get_patient_ds(patient_id, patient_dfs, window_size):
    df = patient_dfs[patient_id]
    if window_size == 0:
        return df.drop(['SepsisLabel', 'ID'], axis=1).values, df['SepsisLabel'].values
    window_dfs = get_patient_windows(df, window_size)
    F = window_dfs[0].shape[1] - 2
    X = np.zeros((len(window_dfs), 6, F))
    y = np.zeros((len(window_dfs), 1))
    for i, w in enumerate(window_dfs):
        X[i], y[i] = get_window_xy(w) 
    return X, y


def get_sepsis_ad_df(df, window_size):
    ids = df['ID'].unique()
    patient_dfs = {pid: df[df['ID'] == pid] for pid in ids}
    patient_dss = [get_patient_ds(p, patient_dfs, window_size) for p in ids]
    Xs, ys = zip(*patient_dss)
    if window_size == 0:
        X = np.stack(Xs, axis=0)
        y = np.stack(ys, axis=0)
    else:
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
    return X, y


def get_pointwise_labels(df):
    ids = df['ID'].unique()
    patient_dfs = {pid: df[df['ID'] == pid] for pid in ids} 
    patient_labels = np.zeros((len(ids), 45))
    for i, (pid, patient_df) in enumerate(patient_dfs.items()):
        patient_labels[i] = patient_df['SepsisLabel'].values
    return patient_labels


def run_sepsis_exp(
    repr_dims,
    epochs,
    model_type='trep',
    window_size=6,
    seed=0,
):
    # Train TRep on Sepsis data
    print(f"Creating and training {model_type} model...")
    if model_type != 'raw_data':
        # hidden_dims = np.array([128, 128, 128, 128, 64, 64, 64, 64, 32, 32])
        task_weights = {
        'local_static_contrast': 0.8,
        'global_vatiant_contrast': 0.1,
        'dynamic_trend_pred': 0.1,
        'dynamic_trend_pred2': 0.,
    }
        # time_embedding = 't2v_sin'

        model = train_trep_sepsis(task_weights)

    # Build datasets for segment-based anomaly detection
    print(f"Loading anomaly detection data...")
    train_df = pd.read_csv(f"{SEPSIS_DATA_PATH}/processed_data/clean_train_df.csv", index_col=0)
    test_df = pd.read_csv(f"{SEPSIS_DATA_PATH}/processed_data/clean_test_df.csv", index_col=0)
    train_X, train_y = get_sepsis_ad_df(train_df, window_size=window_size)
    test_X, test_y = get_sepsis_ad_df(test_df, window_size=window_size)
    print(f"Train X: {train_X.shape}, train y: {train_y.shape}")
    print(f"Test X: {test_X.shape}, test y: {test_y.shape}")
    test_labels = np.zeros((N_TEST_PATIENTS, 45))
    test_labels[:, 5:] = test_y.reshape(-1, 40)

    # Train and test SVM for segment-based anomaly detection
    print(f"Training and testing anomaly detection model...")
    test_preds, test_accs = tasks.eval_anomaly_detection_sepsis(
        model=model if model_type == 'trep' else None,
        train_data=train_X,
        train_labels=train_y.squeeze(),
        test_data=test_X,
        test_labels=test_labels,
        eval_protocol='svm',
        window_size=window_size,
        raw_data=(model_type == 'raw_data')
    )

    return test_accs


if __name__ == "__main__":

    test_accs_trep = run_sepsis_exp(
        repr_dims=32,
        epochs=40,
        model_type='trep',
        window_size=6,
        seed=0
    )
