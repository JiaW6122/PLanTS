import torch
from torch import nn
import torch.nn.functional as F
from utils import max_cross_corr,fast_batch_max_cross_corr

def hierarchical_contrastive_loss(u1,u2,v1,x1,dynamic_pred_task_head,weights,temperature):
    """
    Hierarchical loss.

    Args:
        u1 : Static time-series representations of original time series inputs.
        u2 : Static time-series representations of shuffled time series inputs.
        v1 : Dynamic time-series representations of original time series inputs.
        dynamic_pred_task_head: Task head of conditioned dynamic trends forecasting.
        weights: Loss weights.
    """
    # loss = torch.tensor(0., device=u1.device)
    loss_terms = []
    
    if weights['local_static_contrast'] != 0:
        # loss += weights['local_static_contrast'] * local_static_contrastive_loss(u1, u2)
        # loss = loss+ weights['local_static_contrast'] * local_soft_static_contrastive_loss(u1,x1,temperature)
        loss_terms.append(weights['local_static_contrast'] * local_soft_static_contrastive_loss(u1,x1,temperature))
        # print(weights['local_static_contrast'] * local_soft_static_contrastive_loss(u1,x1,temperature))
        
    if weights['global_vatiant_contrast'] != 0:
        # global_loss = global_variant_constrastive_loss(u1)
        global_loss = global_variant_constrastive_loss_v2_faster(u1,x1,temperature)
        loss_terms.append(weights['global_vatiant_contrast'] * global_loss)
        # loss =loss+ weights['global_vatiant_contrast'] * global_loss
        # print(weights['global_vatiant_contrast'] * global_loss)
    if weights['dynamic_trend_pred'] != 0:
        dynamic_loss = dynamic_cond_pred_loss(u1, v1, dynamic_pred_task_head)
        loss_terms.append(weights['dynamic_trend_pred'] * dynamic_loss)
        # loss =loss+ weights['dynamic_trend_pred'] * dynamic_loss 
        # print(weights['dynamic_trend_pred'] * dynamic_loss)
    # print(loss_terms)

    return sum(loss_terms)



def static_contrastive_loss(u1,u2):
    """
        Static Contrastive loss.

    Args:
        u1 : Static time-series representations of original time series inputs.
        u2 : Static time-series representations of shuffled time series inputs.

    Returns:
        The computed contrastive loss
    """
    B, F_s = u1.size(0), u1.size(1)
    if B==1:
        return u1.new_tensor(0.)
    u = torch.cat([u1,u2], dim=0) # 2B * F_s
    sim = torch.matmul(u, u.t()) # 2B * 2B
    logits = torch.tril(sim, diagonal=-1)[:,:-1] # 2B * (2B-1)
    logits += torch.triu(sim, diagonal=1)[:,1:]
    logits = -F.log_softmax(logits, dim=-1)
    # identify positive pairs
    i = torch.arange(B, device=u1.device)
    loss = (logits[i, B+i-1] + logits[B+i, i])/2
    return loss

def local_static_contrastive_loss(u1,u2):
    """
        Local Static Contrastive loss.

    Args:
        u1 : Static time-series representations of original time series windows. # (batch_size*k, w, out_dims)
        u2 : Static time-series representations of shuffled time series windows.
    """
    u1_static=torch.mean(u1, dim=2) # (batch_size, k, out_dims)
    u2_static=torch.mean(u2, dim=2)
    B, k = u1_static.size(0), u1_static.size(1) 
    if B==1:
        return u1.new_tensor(0.)
    u = torch.cat([u1_static,u2_static], dim=0) # 2B x k x out_d
    u = u.transpose(0, 1) # k x 2B x out_d
    sim = torch.matmul(u, u.transpose(1, 2)) # k x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:,:,:-1] # k x 2B * (2B-1)
    logits += torch.triu(sim, diagonal=1)[:,:,1:]
    logits = -F.log_softmax(logits, dim=-1)
    # identify positive pairs
    i = torch.arange(B, device=u1.device)
    loss = (logits[:, i, B+i-1].mean() + logits[:, B+i, i].mean())/2
    return loss

def local_soft_static_contrastive_loss(u1,x1,temperature):
    """
        Local Soft Static Contrastive loss.

    Args:
        u1 : Static time-series representations of original time series windows. # (batch_size, k, w, out_dims)
        x1 : Original time series windows. # (batch_size, k, w, C)
    """
    B, K, W, O = u1.shape
    B, K, W, C = x1.shape
    u1_reshaped = u1.permute(1, 0, 2, 3) #  K, B, W, O
    x1_reshaped = x1.permute(1, 0, 2, 3)
    S = fast_batch_max_cross_corr(x1_reshaped)  # K x B x B

    weight = torch.tril(S, diagonal=-1)[:, :, :-1]
    weight += torch.triu(S, diagonal=1)[:, :, 1:]
    weight = F.softmax(weight / temperature, dim=-1)

    u1_flat = u1_reshaped.reshape(K, B, W*O)
    # print(u1_flat.shape)
    # print(u1_flat.permute(0, 2, 1).shape)
    sim = torch.matmul(u1_flat, u1_flat.permute(0, 2, 1)) # K x B x B
    # print(sim.shape)

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # K * B * (B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    # print(logits.shape)
    # print(weight.shape)

    logits_reshaped = logits.view(B * K, B-1)  # (B*K, K-1)
    weight_reshaped = weight.view(B * K, B-1)  
    criterion = nn.CrossEntropyLoss()
    # print(logits_reshaped)
    # print(weight_reshaped)
    loss = criterion(logits_reshaped, weight_reshaped)

    return loss

def global_variant_constrastive_loss(u1):
    """
        Static Temporal Contrastive loss.

    Args:
        u1 : Static time-series representations of original time series windows. # (batch_size, k, w, out_dims)
    """
    B, w = u1.size(0), u1.size(2) 
    u_h = u1[:, :, 0:w//2, :] # history windows  (B, k, w//2, out_d)
    u_f = u1[:, :, w//2:, :] # future windows  (B, k, w//2, out_d)
    u_h_static = torch.mean(u_h, dim=2) # B x k x out_d
    u_f_static = torch.mean(u_f, dim=2)

    u = torch.cat([u_h_static,u_f_static], dim=0) # 2B x k x out_d
    u = u.transpose(0, 1) # k x 2B x out_d
    sim = torch.matmul(u, u.transpose(1, 2)) # k x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:,:,:-1] # k x 2B * (2B-1)
    logits += torch.triu(sim, diagonal=1)[:,:,1:]
    logits = -F.log_softmax(logits, dim=-1)
    # identify positive pairs
    i = torch.arange(B, device=u1.device)
    loss = (logits[:, i, B+i-1].mean() + logits[:, B+i, i].mean())/2
    return loss

def global_variant_constrastive_loss_v2(u1, x1, temperature):
    """
        Static Temporal Contrastive loss v2. Using soft regression constrastive loss to deal with false negative problem. See https://github.com/YyzHarry/SimPer?tab=readme-ov-file

    Args:
        u1 : Static time-series representations of original time series windows. # (batch_size, k, w, out_dims)
        x1 : Original time series windows. # (batch_size, k, w, C)
    """
    B, K, W, C = x1.shape
    x_h = x1[:, :, 0:W//2, :] # history windows  (B, k, w//2, C)
    x_f = x1[:, :, W//2:, :] # future windows  (B, k, w//2, C)
    x = torch.cat([x_h, x_f], dim=1) # B x 2k x w//2 x C
    # print("x_shape:",x.shape)
    S = torch.zeros((B, 2*K, 2*K), device= x.device) # Initialize similarity matrices
    for i in range(B):
        # print("sample:",i)
        for p in range(2*K):
            for q in range(p+1, 2*K):
              # print(p)
              # print(q)
              S[i, p, q] = torch.mean(max_cross_corr(x[i, p,: ,:], x[i, q,: ,:]))
              S[i, q, p] = S[i, p, q]
    
    weight = torch.tril(S, diagonal=-1)[:,:,:-1] # B x 2k * (2k-1)
    weight += torch.triu(S, diagonal=1)[:,:,1:]
    weight = F.softmax(weight/temperature, dim=-1)

    u_h = u1[:, :, 0:W//2, :] # history windows  (B, k, w//2, out_d)
    u_f = u1[:, :, W//2:, :] # future windows  (B, k, w//2, out_d)
    u_h_static = torch.mean(u_h, dim=2) # B x k x out_d
    u_f_static = torch.mean(u_f, dim=2)
    u = torch.cat([u_h_static, u_f_static], dim=1)  # B x 2k x out_d
    sim = torch.matmul(u, u.transpose(1, 2))  # B x 2k x 2k
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2k x (2k-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    
    logits_reshaped = logits.view(B * (2*K), 2*K-1)  # (B*2K, 2K-1)
    weight_reshaped = weight.view(B * (2*K), 2*K-1)  # (B*2K, 2K-1)
    criterion = nn.CrossEntropyLoss()
    # print(logits_reshaped)
    # print(weight_reshaped)
    loss = criterion(logits_reshaped, weight_reshaped)

    return loss


def global_variant_constrastive_loss_v2_faster(u1, x1, temperature):
    B, K, W, C = x1.shape
    end=W
    if W%2==1:
        end=W-1
    x_h = x1[:, :, 0:W//2, :]
    x_f = x1[:, :, W//2:end, :]
    x = torch.cat([x_h, x_f], dim=1)  # B x 2K x L x C

    # Efficient similarity matrix
    S = fast_batch_max_cross_corr(x)  # B x 2K x 2K

    weight = torch.tril(S, diagonal=-1)[:, :, :-1]
    weight += torch.triu(S, diagonal=1)[:, :, 1:]
    weight = F.softmax(weight / temperature, dim=-1)

    u_h = u1[:, :, 0:W//2, :]
    u_f = u1[:, :, W//2:end, :]
    u_h_static = torch.mean(u_h, dim=2)
    u_f_static = torch.mean(u_f, dim=2)
    u = torch.cat([u_h_static, u_f_static], dim=1)  # B x 2K x D

    sim = torch.matmul(u, u.transpose(1, 2))  # B x 2K x 2K
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    logits_reshaped = logits.view(B * (2*K), 2*K - 1)
    weight_reshaped = weight.view(B * (2*K), 2*K - 1)

    log_probs = F.log_softmax(logits_reshaped / temperature, dim=-1)
    loss = F.kl_div(log_probs, weight_reshaped, reduction='batchmean')

    return loss




def dynamic_cond_pred_loss(u1,v1,dynamic_pred_task_head):
    '''
    Conditioned dynamic trends forecasting loss. The pretext task is to incorporate the learned static feature representation from the previous time points, to predict the dynamic trends of the future time points.

    Args:
        u1 : Static time-series representations of original time series windows. # (batch_size, k, w, out_dims1)
        v1 : Dynamic time-series representations of original time series windows. # (batch_size, k, w, out_dims2)
    '''
    B, w = u1.size(0), u1.size(2) 
    end=w
    if w%2==1:
        end=w-1
    u_h = u1[:, :, 0:w//2, :] # history windows  (B, k, w//2, out_d1)
    u_f = u1[:, :, w//2:end, :] # future windows  (B, k, w//2, out_d1)

    v_h = v1[:, :, 0:w//2, :] # history windows  (B, k, w//2, out_d2)
    v_f = v1[:, :, w//2:end, :] # future windows  (B, k, w//2, out_d2)

    x = torch.cat((u_h,v_h), dim=3) #(B, k, w//2, out_d1+out_d2)
    y = v_f
    pred=dynamic_pred_task_head(x)
    loss = nn.MSELoss()

    return loss(pred, y)

