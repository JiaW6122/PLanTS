
import argparse
import seaborn as sns
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

from datautils import load_UEA
from utils import init_dl_program, name_with_datetime, pkl_save
from plants import PLanTS
import torch
import gc
import numpy as np
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
import s3fs
import io

from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def parse_args():
    parser = argparse.ArgumentParser(description="Run classification experiment.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (e.g., "ArticularyWordRecognition")')
    parser.add_argument('--run_name', type=str, default='default_run',
                        help='Name of the run for logging, saving, and tracking purposes')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--max_threads', type=int, default=None, help='Maximum number of threads to use')
    parser.add_argument('--static_repr_dims', type=int, default=128, help='Dimension of static representation')
    parser.add_argument('--dynamic_repr_dims', type=int, default=128, help='Dimension of dynamic representation')
    parser.add_argument('--max_train_length', type=int, default=800, help='Maximum training sequence length')
    parser.add_argument('--iters', type=int, default=None, help='Number of iterations')
    parser.add_argument('--distance', type=str, default='mcc', choices=['mcc', 'dwt'], help='Distance metric for similarity')
    parser.add_argument('--window_size', type=str, default='Auto',
                        help='Window size for training. Can be an int (e.g., "10"), a list (e.g., "[50,100,200]"), or a string (e.g., "Auto")')

    parser.add_argument('--top_k', type=int, default=3, help='Number of top periodicities to consider')
    parser.add_argument('--tmp_emb_type', type=str, default="original", choices=["original", 'positional', 'temporal_fixed', 'temporal_learn'], help='Type of temporal embedding')
    parser.add_argument('--freq', type=str, default='h', help='Frequency for temporal embedding')
    parser.add_argument('--weight_local_static_contrast', type=float, default=0.9,
                        help='Weight for the local static contrastive loss')
    parser.add_argument('--weight_global_vatiant_contrast', type=float, default=0.1,
                        help='Weight for the global variant contrastive loss')
    parser.add_argument('--weight_dynamic_trend_pred', type=float, default=0.,
                        help='Weight for the dynamic trend prediction loss')


    return parser.parse_args()


def main(args):

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    

    args.task_weights = {
        'local_static_contrast': args.weight_local_static_contrast,
        'global_vatiant_contrast': args.weight_global_vatiant_contrast,
        'dynamic_trend_pred': args.weight_dynamic_trend_pred,
    }

    try:
        if args.window_size.startswith("[") and args.window_size.endswith("]"):
            args.window_size = list(map(int, args.window_size.strip("[]").split(",")))
        elif args.window_size.isdigit():
            args.window_size = int(args.window_size)
        # else: keep as string (e.g., "full_series")
    except Exception as e:
        raise ValueError(f"Invalid format for --window_size: {args.window_size}") from e


    # train_data, train_labels, test_data, test_labels = datautils.load_UCR("ACSF1")
    # print(f"Shapes - train data: {train_data.shape}, test data: {test_data.shape}")

    
    
    train_data, train_labels, test_data, test_labels = load_UEA(args.dataset_name, s3_bucket="sagemaker-bdrl-aws", s3_prefix="xiaowang/HDST")
    N1, T1, n_channels = train_data.shape
    N2, T2, _ = test_data.shape
    train_mark = np.arange(T1)  
    train_mark_full = np.broadcast_to(train_mark[None, :, None], (N1, T1, 1))  # (N1, T1, 1)
    test_mark = np.arange(T2)  
    test_mark_full = np.broadcast_to(test_mark[None, :, None], (N2, T2, 1))  # (N2, T2, 1)
    print(f"Shapes - train data: {train_data.shape}, test data: {test_data.shape}")
    print(f"Shapes - train tmp: {train_mark_full.shape}, test tmp: {test_mark_full.shape}")
    train_all=np.concatenate((train_data,train_mark_full),axis=2)
    test_all=np.concatenate((test_data,test_mark_full),axis=2)
    print(f"Shapes - train all: {train_all.shape}, test all: {test_all.shape}")

    sns.set_theme()
    torch.cuda.empty_cache()
    gc.collect()

    import time
    import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    start_time = time.time()
    model = PLanTS(
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
        train_all,
        n_epochs=args.epochs,
        n_iters=args.iters,
        w=args.window_size,
        distance=args.distance,
        top_k=args.top_k,
        n_channels=n_channels
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Model training time: {training_time:.2f} seconds")


    run_dir = 'training/' + args.dataset_name + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # import torch
    model.save(f'{run_dir}/model.pkl')
    import pandas as pd
    df = pd.DataFrame({"epoch": list(range(1, len(loss_log) + 1)), "loss": loss_log})
    df.to_csv(f'{run_dir}/loss_log.csv', index=False)


    from tasks.classification import eval_classification
    out, eval_res = eval_classification(
                    model,
                    train_all,
                    train_labels,
                    test_all,
                    test_labels,
                    n_channels,
                    encoding_protocol='full_series',
                    eval_protocol='svm'
                )
    print(eval_res)
    pkl_save(f'{run_dir}/out.pkl', out)
    pkl_save(f'{run_dir}/eval_res.pkl', eval_res)



if __name__ == "__main__":
    args = parse_args()
    main(args)
