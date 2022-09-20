import numpy as np
import random
from random import randint
from utils.data_utils import rotmat2euler_torch, expmap2rotmat_torch

import torch
import torch.nn as nn


def criterion_cos(input_f, target_f):
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    return cos(input_f, target_f)


def criterion_cos2(input_f, target_f):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(input_f, target_f)



def mpjpe_error(batch_pred,batch_gt): 
    
    batch_pred= batch_pred.contiguous().view(-1,3)
    batch_gt=batch_gt.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))


def euler_error(ang_pred, ang_gt):
    # only for 32 joints
    dim_full_len=ang_gt.shape[2]

    pred_expmap = ang_pred.contiguous().view(-1,dim_full_len).view(-1, 3)
    targ_expmap = ang_gt.contiguous().view(-1,dim_full_len).view(-1, 3)

    pred_eul = rotmat2euler_torch(expmap2rotmat_torch(pred_expmap))
    pred_eul = pred_eul.view(-1, dim_full_len)

    targ_eul = rotmat2euler_torch(expmap2rotmat_torch(targ_expmap))
    targ_eul = targ_eul.view(-1, dim_full_len)
    mean_errors = torch.mean(torch.norm(pred_eul - targ_eul, 2, 1))

    return mean_errors


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m



def get_dct_in (input_seq):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dct_used = input_seq.shape[1]
    dct_m_in, _ = get_dct_matrix(dct_used)
    
    dct_m_in = torch.from_numpy(dct_m_in.astype('float32')).to(device)
    
    input_dct_seq = torch.matmul(dct_m_in[:, 0:dct_used], input_seq)
    
    return input_dct_seq


def get_dct_out (input_seq):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dct_used = input_seq.shape[1]
    _, idct_m_in = get_dct_matrix(dct_used)
    
    idct_m_in = torch.from_numpy(idct_m_in.astype('float32')).to(device)
    
    input_dct_seq = torch.matmul(idct_m_in[:, 0:dct_used], input_seq)
    
    return input_dct_seq




# def delta_2_gt (prediction, last_timestep):
#     prediction = prediction.clone()
    
#     #print (prediction [:,0,:].shape,last_timestep.shape)
#     prediction [:,0,:] = prediction [:,0,:] + last_timestep
#     prediction [:,1,:] = prediction [:,1,:] + prediction [:,0,:]
    
#     prediction [:,2,:] = prediction [:,2,:] + prediction [:,1,:]    
#     prediction [:,3,:] = prediction [:,3,:] + prediction [:,2,:]
    
#     prediction [:,4,:] = prediction [:,4,:] + prediction [:,3,:]
#     prediction [:,5,:] = prediction [:,5,:] + prediction [:,4,:]
#     prediction [:,6,:] = prediction [:,6,:] + prediction [:,5,:]
#     prediction [:,7,:] = prediction [:,7,:] + prediction [:,6,:]
    
#     prediction [:,8,:] = prediction [:,8,:] + prediction [:,7,:]
#     prediction [:,9,:] = prediction [:,9,:] + prediction [:,8,:]
    
#     prediction [:,10,:] = prediction [:,10,:] + prediction [:,9,:]
#     prediction [:,11,:] = prediction [:,11,:] + prediction [:,10,:]
#     prediction [:,12,:] = prediction [:,12,:] + prediction [:,11,:]
#     prediction [:,13,:] = prediction [:,13,:] + prediction [:,12,:]
    
#     prediction [:,14,:] = prediction [:,14,:] + prediction [:,13,:]
#     prediction [:,15,:] = prediction [:,15,:] + prediction [:,14,:]
#     prediction [:,16,:] = prediction [:,16,:] + prediction [:,15,:]
#     prediction [:,17,:] = prediction [:,17,:] + prediction [:,16,:]
    
#     prediction [:,18,:] = prediction [:,18,:] + prediction [:,17,:]
#     prediction [:,19,:] = prediction [:,19,:] + prediction [:,18,:]
#     prediction [:,20,:] = prediction [:,20,:] + prediction [:,19,:]
#     prediction [:,21,:] = prediction [:,21,:] + prediction [:,20,:]
    
#     prediction [:,22,:] = prediction [:,22,:] + prediction [:,21,:]
#     prediction [:,23,:] = prediction [:,23,:] + prediction [:,22,:]
#     prediction [:,24,:] = prediction [:,24,:] + prediction [:,23,:]
    
    
    
#     # for i in range (args.output_n -1):
#     #     prediction [:,i+1,:] = prediction [:,i+1,:] + prediction [:,0,:]
        
#     return prediction



def delta_2_gt (prediction, last_timestep):
    prediction = prediction.clone()
    
    #print (prediction [:,0,:].shape,last_timestep.shape)
    prediction [:,0,:] = prediction [:,0,:] + last_timestep
    for i in range (prediction.shape[1]-1):
        prediction [:,i+1,:] = prediction [:,i+1,:] + prediction [:,i,:]


        
    return prediction




def mask_sequence (seq,mframes):
    
    x = [randint(0, seq.shape[1]-1) for p in range(0, mframes)]
    
    for i in x:
        seq[:,i,:] = 0
        
    return seq 



def mask_joints (seq,mjoints):
    
    seq_masked = seq.clone()
    #x = [randint(0, seq.shape[1]-1) for p in range(0, 22) if p % 3 == 0 ]
    x = [random.randrange(0, 66, 3) for p in range(0, mjoints)]
    
    for i in x:
        seq_masked[:,:,i] = 0
        seq_masked[:,:,i+1] = 0
        seq_masked[:,:,i+2] = 0
        
    return seq_masked
    
    
    
    
    
    
    
    