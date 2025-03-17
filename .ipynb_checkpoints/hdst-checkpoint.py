import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.encoder import HDEncoder2
from utils import split_with_nan, centerize_vary_length_series,  extract_fixed_random_windows, torch_pad_with, torch_pad_nan
from models.task_heads import DynamicCondPredHead
from models.losses import hierarchical_contrastive_loss
import time

class HDST:
    '''The HDST model'''

    def __init__(
        self,
        input_dims,
        kernels1=3,
        kernels2=3,
        output_dims1=128,
        output_dims2=128,
        hidden_dims1=np.array([64, 128, 256, 256, 256, 256, 128, 128, 128, 128]),
        hidden_dims2=np.array([64, 128, 256, 256, 256, 256, 128, 128, 128, 128]),
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        task_weights=None,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a HDST model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims1 (int): The static representation dimension.
            output_dims2 (int): The dynamic representation dimension.
            hidden_dims1 (int): The hidden dimension of the static encoder.
            hidden_dims2 (int): The hidden dimension of the dynamic encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self._net=HDEncoder2(input_dims=input_dims,
                            output_dims1=output_dims1,
                            output_dims2=output_dims2,
                            kernels1=kernels1,
                            kernels2=kernels2,
                            hidden_dims1=hidden_dims1,
                            hidden_dims2=hidden_dims2, 
                            depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        if task_weights is not None:
            self.task_weights = task_weights
        else:
            self.task_weights = {
                'local_static_contrast': 0.33,
                'global_vatiant_contrast': 0.33,
                'dynamic_trend_pred': 0.34,
            }
        assert sum(self.task_weights.values()) == 1.0
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.dynamic_pred_task_head = DynamicCondPredHead(
            in_features=output_dims1 + output_dims2,
            hidden_features=[64,128],
            out_features=output_dims2,
        ).to(self.device)

        self.n_epochs = 0
        self.n_iters = 0


    def fit(self, train_data, k, w, temperature=1, n_epochs=None, n_iters=None):
        ''' Training the HDST model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            k (int) : Number of windows that randomly cropped while training.
            w (int) : Number of time point of each window (window size).
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        
        # Split data into windows, pad windows with nans to have equal lengths
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        # What timesteps have no modalities present for at least one batch element
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        # Eliminate empty series        
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
        print(f"Training data shape: {train_data.shape}")
        
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        train_start = time.time()
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                # Batch is a 1 element list
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                
                # randomly choose k windows of size w(batch-wise)
                # print(x.shape)
                x_windows=extract_fixed_random_windows(x, w, k) # (batch_size, k, w, n_features)
                # print("x_windows:")
                # print(x_windows.shape)
                B, k, w, nf=x_windows.shape
                x_windows_reshaped = x_windows.reshape(B * k, w, nf)  

                # original windows
                out_static, out_dynamic=self._net(x_windows_reshaped) # (batch_size*k, w, out_dims)
                out_dim1 = out_static.size(-1)
                out_dim2 = out_dynamic.size(-1)
                out_static = out_static.reshape(B, k, w, out_dim1)
                out_dynamic = out_dynamic.reshape(B, k, w, out_dim2)

                # shuffled windows
                shuffle_idx = torch.randperm(x_windows.shape[2])
                # consider shuffle each window using diffent shuffle_idx!!!!!!!!!!!!!!!!!!!!!
                shuffled_x_windows = x_windows_reshaped[:,shuffle_idx,:].contiguous()
                out_shuffled_static, _ = self._net(shuffled_x_windows)

                out_shuffled_static = out_shuffled_static.reshape(B, k, w, out_dim1)

                optimizer.zero_grad()
                
                loss = hierarchical_contrastive_loss(
                    out_static,
                    out_shuffled_static,
                    out_dynamic,
                    x_windows,
                    dynamic_pred_task_head = self.dynamic_pred_task_head,
                    weights = self.task_weights,
                    temperature = temperature
                )
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            
            print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

        return loss_log

    def _eval_with_pooling(
            self,
            x,
            slicing=None,
            encoding_window=None,
        ):
        out_static, out_dynamic = self.net(x.to(self.device, non_blocking=True))
        out = torch.cat([out_static, out_dynamic], dim=-1)
        
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = encoding_window // 2,
                padding = encoding_window // 2
            ).transpose(1, 2)

            if encoding_window % 2 == 0:
                out = out[:, :-1]
                
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                
                if slicing is not None:
                    t_out = t_out[:, slicing]
                    
                reprs.append(t_out)
                
                p += 1
            out = torch.cat(reprs, dim=-1)
            
            
        else:
            if slicing is not None:
                out = out[:, slicing]
                
            
        return out.cpu()
    
    def encode(
            self,
            data,
            encoding_window=None,
            causal=False,
            sliding_length=None,
            sliding_padding=0,
            batch_size=None
        ):
        ''' Compute representations using the model.
        
        Args:
            data (np.ndarray): This should have a shape of (n_instances, n_timestamps, n_features). All missing data should be set to NaN.
            time_indices (np.ndarray): Timestep indices to be fed to the time-embedding module. The 'find_closest_train_segment' from tasks.forecasting.py can be used to find timesteps at which the test set most resembles the train set.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'. It is used for anomaly detection, otherwise left to 'None'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation undergoes max pooling with kernel size determined by this param. It can be set to 'full_series' (Collapsing the time dimension of the representation to 1), 'multiscale' (combining representations at different time-scales) or an integer specifying the pooling kernel size. Leave to 'None' and no max pooling will be applied to the time dimension: it will be the same as the raw data's.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp. This is done using causal convolutions.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            return_time_embeddings (bool): Whether to only return the encoded time-series representations, or also the associated time-embedding vectors.
            
        Returns:
            repr, time_embeddings: 'repr' designates the representations for the input time series. The representation's associated time-embeddings are also returned if 'return_time_embeddings' is set to True.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
 
        with torch.no_grad():
            output = []
            output_time_embeddings = []
            for batch in loader:

                
                x = batch[0]

                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        

                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window,
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                                
                            calc_buffer.append(x_sliding)
                            
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window,
                            )
                            reprs.append(out)
                            

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window,
                            )
                            reprs += torch.split(out, n_samples)
                            
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                        
                else:
                    out = self._eval_with_pooling(
                        x,
                        encoding_window=encoding_window,
                    )
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                        
                output.append(out)
                
                
            output = torch.cat(output, dim=0)
            
            
        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)













  

        
        