import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
from functools import partial
from einops import reduce, rearrange, repeat
from torch import distributions as dist
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        n, t, c2 = x.shape
        res = self.pe[:, :x.size(1)]
        res=res.expand(n,res.shape[1],res.shape[2])
        return res

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, out_dims, hidden_dims, embed_type='temporal_fixed', freq='h'):
        '''
        embed_type: Define the embedding type of time stamps. 
        'temporal_fixed': using FixedEmbedding with explicit timestamps inputs.
        'temporal_learn': using nn.Embedding with explicit timestamps inputs.
        'positional': using PositionalEmbedding with synthetic timestamps inputs.
        '''
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        if embed_type == 'temporal_fixed':
            Embed = FixedEmbedding
        elif embed_type == 'temporal_learn':
            Embed = nn.Embedding
        self.embed_type = embed_type

        # Embed = FixedEmbedding if embed_type == 'temporal_fixed' else nn.Embedding
        if self.embed_type == 'temporal_fixed' or self.embed_type == 'temporal_learn':
            if freq == 't':
                self.minute_embed = Embed(minute_size, d_model)
            self.hour_embed = Embed(hour_size, d_model)
            self.weekday_embed = Embed(weekday_size, d_model)
            self.day_embed = Embed(day_size, d_model)
            self.month_embed = Embed(month_size, d_model)
        else:
            self.postional_embed = PositionalEmbedding(d_model)

        layers = []
        input_dim = d_model
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, out_dims))  
        self.post_mlp = nn.Sequential(*layers)

        self.input_fc = nn.Linear(d_model, hidden_dims[0])

        self.dilated_conv=DilatedConvEncoder(
                hidden_dims[0],
                list(hidden_dims) + [out_dims],
                kernel_size=3)
        self.dropout=nn.Dropout(p=0.1)

    def forward(self, x):
        # print('input:',x.shape)
        x = x.long()
        # print(x.shape)
        # print("Hour values: ", x[:, :, 3].min(), x[:, :, 3].max())
        if self.embed_type == 'temporal_fixed' or self.embed_type == 'temporal_learn':
            minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
            hour_x = self.hour_embed(x[:, :, 3])
            weekday_x = self.weekday_embed(x[:, :, 2])
            day_x = self.day_embed(x[:, :, 1])
            month_x = self.month_embed(x[:, :, 0])
            x = hour_x + weekday_x + day_x + month_x + minute_x
        else:
            x = self.postional_embed(x)

        
        # x = self.post_mlp(x)
        x = self.input_fc(x)
        x = x.transpose(1, 2)
        x = self.dropout(self.dilated_conv(x))
        x = x.transpose(1, 2)

        return x

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

#Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1


        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 1, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes] = compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        #Return to time domain
        x = torch.irfft(out_ft, 1, normalized=True, onesided=True, signal_sizes=(x.size(-1), ))
        return x

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, output_size, use_hidden=True, bidirect=False):
        super().__init__()
        self.main_net = nn.LSTM(input_size=input_size, hidden_size=output_size, bidirectional=bidirect)
        self.use_hidden=use_hidden
        self.bidirect=bidirect

    def forward(self, x):
        outs, hidden, cell = self.main_net(x)
        if self.use_hidden:
            return hidden
        else:
            return cell


class SimpleBlock1d(nn.Module):
    def __init__(self, modes, hidden_dims2, component_dims):
        super(SimpleBlock1d, self).__init__()

        self.modes1 = modes
        self.width = hidden_dims2
        self.width2 = component_dims

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm1d(self.width)
        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)


        self.fc1 = nn.Linear(self.width, self.width2)

    def forward(self, x):

        # x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        
        return x

class TimeInvariantEncoder(nn.Module):
    def __init__(self, input_size=64, z_size=64, static=False):
        super().__init__()
        self.main_net = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1)
        self.mean = nn.Linear(in_features=128 * (1 + int(static)), out_features=z_size)
        self.std = nn.Linear(in_features=128 * (1 + int(static)), out_features=z_size)
        self.static = static

    def forward(self, encoded):
        outs, hidden = self.main_net(encoded)
        if self.static:
            hidden = torch.cat(hidden, 2).squeeze(0)
            mean = self.mean(hidden)
            std = F.softplus(self.std(hidden))
        else:
            mean = self.mean(outs)
            std = F.softplus(self.std(outs))
        return dist.Normal(loc=mean, scale=std)


class HDEncoder0(nn.Module):
    def __init__(self, input_dims, output_dims, kernels, modes, hidden_dims1=32, hidden_dims2=64, hidden_dims3=128, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.kernels=kernels
        self.hidden_dims1 = hidden_dims1
        self.hidden_dims2 = hidden_dims2
        self.mask_mode = mask_mode
        component_dims= (output_dims-hidden_dims2)//2
        self.component_dims = component_dims
        self.signal_extractor=nn.Module([
            nn.Linear(input_dims,hidden_dims1),
            nn.Linear(hidden_dims1,hidden_dims2)
        ])
        self.noice_extractor=nn.Module([
            nn.Linear(input_dims,hidden_dims1),
            nn.Linear(hidden_dims1,hidden_dims2)
        ])
        self.time_invariant_encoder=TimeInvariantEncoder(hidden_dims2,hidden_dims2)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims2,
            [hidden_dims2] * depth + [hidden_dims3],
            kernel_size=3
        )
        self.trend_extractor=nn.ModuleList(
            [nn.Conv1d(hidden_dims3, component_dims, k, padding=k-1) for k in kernels]
        )
        self.season_extractor=SimpleBlock1d(modes, hidden_dims3, component_dims)
        
    def forward(self, x, other, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        signal = self.signal_extractor(x)  # B x T x hidden_dims2
        noice = self.noice_extractor(x)   # B x T x hidden_dims2
        
        static = self.time_invariant_encoder(noice)
        # shuffle batch to get positive samples
        shuffle_idx = torch.randperm(noice.shape[1])
        shuffled_encoded_tensor = noice[:, shuffle_idx].contiguous()
        static_pos = self.time_invariant_encoder(shuffled_encoded_tensor)

        # get negative samples from another data
        another_encoded_tensor = self._encode_net(other).transpose(0, 1).contiguous()
        static_neg = self.time_invariant_encoder(another_encoded_tensor)

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(signal.size(0), signal.size(1)).to(signal.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(signal.size(0), signal.size(1)).to(signal.device)
        elif mask == 'all_true':
            mask = signal.new_full((signal.size(0), signal.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = signal.new_full((signal.size(0), signal.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = signal.new_full((signal.size(0), signal.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        signal[~mask] = 0
        
        # conv encoder
        signal = signal.transpose(1, 2)  # B x hidden_dims2 x T
        signal = self.feature_extractor(signal)  # B x hidden_dims3 x T

        trend = []
        for idx, mod in enumerate(self.trend_extractor):
            out = mod(signal)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # b t d
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        season = self.season_extractor(signal)

        return trend, season, static, static_pos, static_neg

class HDEncoder1(nn.Module):
    def __init__(self, input_dims, output_dims, kernels, modes, hidden_dims1=32, hidden_dims2=64, static_dim=128, hidden_dims3=128, depth=10):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.kernels=kernels
        self.hidden_dims1 = hidden_dims1
        self.hidden_dims2 = hidden_dims2
        self.static_dim = static_dim
        self.static_extractor=nn.Module([
            nn.Linear(input_dims,hidden_dims1),
            nn.Linear(hidden_dims1,hidden_dims2),
            LSTMEncoder(hidden_dims2,static_dim)
        ])
        self.dynamic_extractor=nn.Module([
            nn.Linear(input_dims,hidden_dims1),
            nn.Linear(hidden_dims1,hidden_dims2),
            DilatedConvEncoder(
            hidden_dims2,
            [hidden_dims2] * depth + [hidden_dims3],
            kernel_size=3
        )
        ])
    def forward(self, x, x_shuffle): # x: B x T x input_dims
        # x
        static=self.static_extractor(x) # B x static_dim

        # positive sample
        static_pos=self.static_extractor(x_shuffle)
        
        return static, static_pos

class HDEncoder2(nn.Module):
    def __init__(self, input_dims, output_dims1, output_dims2, kernels1, kernels2, tmp_emb_type, freq='h', hidden_dims1=32, hidden_dims2=64, depth=10,mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims1 = output_dims1 #output dims of static features
        self.kernels1=kernels1 #kernels of static encoder
        self.hidden_dims1 = hidden_dims1 #hidden layers dims of static features
        
        self.output_dims2 = output_dims2 #output dims of dynamic features
        self.kernels2=kernels2 #kernels of dynamic encoder
        self.hidden_dims2 = hidden_dims2 #hidden layers dims of dynamic features

        self.mask_mode = mask_mode
        self.tmp_emb_type = tmp_emb_type
        
        # static encoder
        fc_dim = hidden_dims1 if isinstance(hidden_dims1, int) else hidden_dims1[0]
        self.input_fc = nn.Linear(input_dims, fc_dim)

        
        if isinstance(hidden_dims1, np.ndarray):
            assert depth == len(hidden_dims1)
            self.static_feature_extractor = DilatedConvEncoder(
                hidden_dims1[0],
                list(hidden_dims1) + [output_dims1],
                kernel_size=kernels1
            )
        else:
            self.static_feature_extractor = DilatedConvEncoder(
                hidden_dims1,
                [hidden_dims1] * depth + [output_dims1],
                kernel_size=kernels1
            )
        self.repr_dropout = nn.Dropout(p=0.1)

        
        # self.tmp_embedding=TemporalEmbedding(d_model=self.output_dims1)
        if tmp_emb_type=='original':
            # dynamic encoder
            fc_dim2 = hidden_dims2 if isinstance(hidden_dims2, int) else hidden_dims2[0]
            self.input_fc2 = nn.Linear(input_dims, fc_dim2)

            if isinstance(hidden_dims2, np.ndarray):
                assert depth == len(hidden_dims2)
                self.dynamic_feature_extractor = DilatedConvEncoder(
                    hidden_dims2[0],
                    list(hidden_dims2) + [output_dims2],
                    kernel_size=kernels2
                )
            else:
                self.dynamic_feature_extractor = DilatedConvEncoder(
                    hidden_dims2,
                    [hidden_dims2] * depth + [output_dims2],
                    kernel_size=kernels2
                )
            self.repr_dropout2 = nn.Dropout(p=0.1)
        else:
            self.tmp_embedding = TemporalEmbedding(
                                        d_model=128,
                                        out_dims=self.output_dims1,
                                        embed_type=tmp_emb_type,
                                        freq=freq,
                                        hidden_dims=[64, 128, 128])
        
    def forward(self, x,x_tmp,mask=None): # x: (B * k) x w x input_dims
        # x=x_all[:,:,:n_channels]
        # x_tmp=x_all[:,:,n_channels:]
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x_tmp[~nan_mask] = 0

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        #static feature representations
        x_static = self.input_fc(x)  # (B * k) x w x fc_dims
        # conv encoder
        x_static = x_static.transpose(1, 2)  # (B * k) x fc_dims x w
        x_static = self.repr_dropout(self.static_feature_extractor(x_static))  # (B * k) x out_dims1 x w
        x_static = x_static.transpose(1, 2)  # (B * k) x w x out_dims1

        
        if self.tmp_emb_type == 'original':
            #dynamic feature representations
            x_dynamic = self.input_fc2(x)  # (B * k) x w x fc_dims2
            # conv encoder
            x_dynamic = x_dynamic.transpose(1, 2)  # (B * k) x fc_dims2 x w
            x_dynamic = self.repr_dropout2(self.dynamic_feature_extractor(x_dynamic))  # (B * k) x out_dims2 x w
            x_dynamic = x_dynamic.transpose(1, 2)  # (B * k) x w x out_dims2
            # print(x_dynamic.shape)
        else:
            tmp_emb=self.tmp_embedding(x_tmp)
            x_dynamic=tmp_emb
        
        return x_static, x_dynamic