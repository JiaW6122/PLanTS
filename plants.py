import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from models.encoder import PlanTSEncoder
from utils import split_with_nan, centerize_vary_length_series,  extract_fixed_random_windows, torch_pad_with, torch_pad_nan, FFT_for_Period, pad_nan_to_target
from models.task_heads import DynamicCondPredHead, DynamicConstructHead, DynamicPredHead, DynamicConstructPredHead
from models.losses import hierarchical_contrastive_loss
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class PLanTS:
    '''The PLanTS model'''

    def __init__(
        self,
        input_dims,
        tmp_embed_type,
        freq='h',
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
        mask_mode='binomial',
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a PLanTS model.
        
        Args:
            input_dims (int): The input dimension. 
            output_dims1 (int): The latent state representation dimension.
            output_dims2 (int): The dynamic transition representation dimension.
            hidden_dims1 (int): The hidden dimension of the latent state encoder.
            hidden_dims2 (int): The hidden dimension of the dynamic transition encoder.
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

        self._net=PlanTSEncoder(input_dims=input_dims,
                            output_dims1=output_dims1,
                            output_dims2=output_dims2,
                            kernels1=kernels1,
                            kernels2=kernels2,
                            tmp_emb_type=tmp_embed_type,
                            freq=freq,
                            hidden_dims1=hidden_dims1,
                            hidden_dims2=hidden_dims2, 
                            depth=depth,
                            mask_mode=mask_mode).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        if task_weights is not None:
            self.task_weights = task_weights
        else:
            self.task_weights = {
                'local_static_contrast': 0.25,
                'global_vatiant_contrast': 0.25,
                'dynamic_trend_pred': 0.25,
            }
        assert sum(self.task_weights.values()) == 1.0
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.dynamic_pred_task_head = DynamicCondPredHead(
            in_features=output_dims1 + output_dims2,
            hidden_features=[64,128],
            out_features=output_dims2,
        ).to(self.device)
        self.dynamic_pred_task_head2 = DynamicCondPredHead(
            in_features=output_dims1 + output_dims2,
            hidden_features=[64,128],
            out_features=input_dims,
        ).to(self.device)

        self.dynamic_pred_task_head3 = DynamicCondPredHead(
            in_features=output_dims1 + output_dims2,
            hidden_features=[64,128],
            out_features=output_dims1 + output_dims2,
        ).to(self.device)

        self.dynamic_construct_head = DynamicConstructHead(
            in_features=2*output_dims1,
            hidden_features=[128],
            out_features=input_dims,
        ).to(self.device)

        self.dynamic_pred_head = DynamicPredHead(
            in_features=2*output_dims1,
            hidden_features=[128],
            out_features=input_dims,
        ).to(self.device)

        self.n_epochs = 0
        self.n_iters = 0


    def fit(self, train_all, n_channels, distance='mcc', w=None,top_k=5, temperature=1, n_epochs=None, n_iters=None):
        ''' Training the PlanTS model.
        Reshaped the input batch into shape (B, k, scale, C), where k*scale == n_timestampes.
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            distance (str): The distance metric used in contrastive loss. It can be set to 'mcc' (maximum cross-correlation) or 'dtw' (dtw distance).
            w (int) : Number of time point of each window (window size). Using multi-scale if not specified.
            top_k (int) : Numer of top scales
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_all.ndim == 3
        train_data0=train_all[:,:,:n_channels]
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data0.size <= 100000 else 600  # default param for n_iters
        
        # Split data into windows, pad windows with nans to have equal lengths
        if self.max_train_length is not None:
            sections = train_all.shape[1] // self.max_train_length
            if sections >= 2:
                train_all = np.concatenate(split_with_nan(train_all, sections, axis=1), axis=0)

        # What timesteps have no modalities present for at least one batch element
        temporal_missing = np.isnan(train_all).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_all = centerize_vary_length_series(train_all)

        # Eliminate empty series        
        train_all = train_all[~np.isnan(train_all).all(axis=2).all(axis=1)]
        
        train_data=train_all[:,:,:n_channels]
        print(f"Training data shape: {train_data.shape}")


        #### multi-scale using FFT periods
        if w=='Auto':
            scale_list, scale_weight = FFT_for_Period((torch.from_numpy(train_data).to(torch.float)), top_k)
            print(f"Scale list: {scale_list}")
            top_k=len(scale_list)
        if isinstance(w, int):
            scale_list=[w]
            print(f"Scale list: {scale_list}")
            # scale_weight=[1]
            top_k=1
        if isinstance(w, list) and all(isinstance(x, int) for x in w):
            scale_list=w
            print(f"Scale list: {scale_list}")
            top_k=len(w)
            

        
        # train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_dataset_all = TensorDataset(torch.from_numpy(train_all).to(torch.float))

        # print(len(train_dataset))
        train_loader = DataLoader(train_dataset_all, batch_size=min(self.batch_size, len(train_dataset_all)), shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # lr scheduler
        torch.autograd.set_detect_anomaly(True)
        
        loss_log = []
        train_start = time.time()
        total_steps = n_epochs 
        # print(len(train_loader))
        progress_bar = tqdm(total=total_steps, desc="Training")
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            # multi_scale_loss=0
            
            interrupted = False

            
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                # Batch is a 1 element list
                x = batch[0]
                # print(x.shape)
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]

                # if w==None:
                #     # Identify scales
                #     scale_list, scale_weight = FFT_for_Period(x, top_k)
                #     # print(scale_list)
                #     # print(type(scale_list[0]))
                # else:
                #     scale_list=[w]
                #     scale_weight=[1]
                #     top_k=1
                x = x.to(self.device)
                loss=list()
                multi_scale_loss=0
                # print(scale_list)

                for i in range(top_k):
                    scale = scale_list[i]
                    if scale>self.max_train_length:
                        scale=self.max_train_length
                    # print(type(scale))
                    # print(type(x))
                    B, T, C2 = x.shape
                    if T%scale!=0:
                        length=(T//scale+1)*scale
                        x=torch_pad_nan(x,left=0,right=length-T,dim=1)
                    else:
                        length=T

                    k=length//scale
                    x_windows=x.reshape(B, k, scale, C2)
                    # print("x_windows:")
                    # print(x_windows.shape)
                    # x_windows_reshaped = x_windows.reshape(B * k, scale, C2) 
                    x_data=x_windows[:,:,:,:n_channels]
                    tmp_stamp=x_windows[:,:,:,n_channels:]
                    

                    x_data_reshaped=x_data.reshape(B * k, scale, n_channels)
                    tmp_stamp_reshape=tmp_stamp.reshape(B * k, scale, C2-n_channels)

    
                    # original windows
                    out_static, tmp_embed=self._net(x_data_reshaped,tmp_stamp_reshape) # (batch_size*k, scale, out_dims)
                    # print(out_static.shape)
                    # print(tmp_embed.shape)
                    out_dim1 = out_static.size(-1)
                    out_dim2 = tmp_embed.size(-1)
                    out_static = out_static.reshape(B, k, scale, out_dim1)
                    tmp_embed = tmp_embed.reshape(B, k, scale, out_dim2)
    
    
                    optimizer.zero_grad()
                    
                    loss_scale = hierarchical_contrastive_loss(
                        out_static,
                        tmp_embed,
                        x_data,
                        distance,
                        dynamic_pred_task_head = self.dynamic_pred_task_head,
                        weights = self.task_weights,
                        temperature = temperature
                    )
                    multi_scale_loss= multi_scale_loss+loss_scale
                    loss.append(loss_scale)
                    loss_scale.backward()
                    optimizer.step()
                    self.net.update_parameters(self._net)

                # print(loss)
                multi_scale_loss = multi_scale_loss / top_k
                    
                cum_loss =cum_loss + multi_scale_loss.item()
                # cum_loss =cum_loss + sum(loss)/top_k
                n_epoch_iters += 1

                # print(f"n_epoch_iters #{n_epoch_iters}: loss={loss.item()}")
                
                self.n_iters += 1
            
            if interrupted:
                break
            
            cum_loss =cum_loss/n_epoch_iters
            loss_log.append(cum_loss)

            

            progress_bar.set_postfix(loss=cum_loss, epoch=self.n_epochs+1)
            progress_bar.update(1)
            # print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            # scheduler.step()

        progress_bar.close()

        return loss_log

    def _eval_with_pooling(
            self,
            x,
            n_channels,
            mask=None,
            slicing=None,
            encoding_window=None,
        ):
        # print(x.shape)
        x_data=x[:,:,:n_channels]
        x_tmp=x[:,:,n_channels:]
        out_static, tmp_embed = self.net(x_data.to(self.device, non_blocking=True),x_tmp.to(self.device, non_blocking=True),mask)
        out = torch.cat([out_static, tmp_embed], dim=-1)
        
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
        # return out

    
    def encode(
            self,
            data_all,
            n_channels,
            batch_size=None,
            mask=None,
            encoding_window=None,
            causal=False,
            sliding_length=None,
            sliding_padding=0,
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
        assert data_all.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data_all.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data_all).to(torch.float))
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
                                    n_channels,
                                    mask=mask,
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
                                n_channels,
                                mask=mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window,
                            )
                            reprs.append(out)
                            

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                n_channels,
                                mask=mask,
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
                        n_channels,
                        mask=mask,
                        encoding_window=encoding_window,
                    )
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                        
                output.append(out)
                
            # print(output[0].shape)
            # print(len(output))
            # output = [o.numpy() for o in output]  # each o is shape (8, 1000, 256)
            # output = np.concatenate(output, axis=0)

            output = torch.cat(output, dim=0)
            # output = output.cpu().numpy()
            torch.cuda.empty_cache()
            
            
        self.net.train(org_training)
        return output.numpy()
        # return output

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

    def eval(self):
        self.net.eval()













  

        
        