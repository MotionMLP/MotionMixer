import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.autograd
import matplotlib.pyplot as plt
from utils.ang2joint import *
from dataloader_amass import *
import numpy as np
import argparse
import os
from mlp_mixer import MlpMixer




def test_mixer(model, args):

    device = args.dev
    model.eval()
    accum_loss = 0
    n_batches = 0  # number of batches for all the sequences

    n = 0
       

    Dataset = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=2)
    loader_test = DataLoader( Dataset,batch_size=args.batch_size,
                              shuffle =False,num_workers=0)
        
                      
    joint_used=np.arange(4,22)
    full_joint_used=np.arange(0,22) # needed for visualization
    with torch.no_grad():
        for cnt,batch in enumerate(loader_test): 
            batch = batch.float().to(device)
            batch_dim=batch.shape[0]
            n+=batch_dim
            
            sequences_train=batch[:,0:args.input_n,joint_used,:].view(-1,args.input_n,args.pose_dim)
    
            sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,full_joint_used,:]#.view(-1,args.output_n,args.pose_dim)
            
            sequences_predict=model(sequences_train).view(-1,args.output_n,18,3)#.permute(0,1,3,2)
            
            
            all_joints_seq=sequences_predict_gt.clone()
    
            all_joints_seq[:,:,joint_used,:]=sequences_predict
    
            loss=mpjpe_error(all_joints_seq,sequences_predict_gt)*1000 # loss in milimeters
            accum_loss+=loss*batch_dim
    print('overall average loss in mm is: '+str(accum_loss/n))



    return accum_loss/n_batches



if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False) # Parameters for mpjpe
    parser.add_argument('--data_dir', type=str, default='../data_amass/', help='path to the unziped dataset directories(H36m/AMASS/3DPW)')  
    parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
    parser.add_argument('--output_n', type=int, default=25, help="number of model's output frames")
    parser.add_argument('--skip_rate', type=int, default=5, choices=[1, 5], help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')
    parser.add_argument('--root', default='./runs', type=str, help='root path for the logging') 

    parser.add_argument('--activation', default='gelu', type=str, required=False) 
    parser.add_argument('--r_se', default=8, type=int, required=False)

    parser.add_argument('--n_epochs', default=50, type=int, required=False)
    parser.add_argument('--batch_size', default=50, type=int, required=False) 
    parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
    parser.add_argument('--pin_memory', default=False, type=bool, required=False)
    parser.add_argument('--loader_workers', default=4, type=int, required=False)
    parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)
    parser.add_argument('--dev', default='cuda:0', type=str, required=False)
    parser.add_argument('--initialization', type=str, default='none', help='none, glorot_normal, glorot_uniform, hee_normal, hee_uniform')
    parser.add_argument('--use_scheduler', default=True, type=bool, required=False)
    parser.add_argument('--milestones', type=list, default=[15, 25, 35, 40], help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma correction to the learning rate, after reaching the milestone epochs')
    parser.add_argument('--clip_grad', type=float, default=None, help='select max norm to clip gradients')
    parser.add_argument('--model_path', type=str, default='./checkpoints/amass_3d_25frames_ckpt', help='directory with the models checkpoints ')
    parser.add_argument('--batch_size_test', type=int, default=256, help='batch size for the test set')
    parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'], help='choose data split to visualize from(train-val-test)')

    args = parser.parse_args()

    parser_mpjpe = argparse.ArgumentParser(parents=[parser]) # Parameters for mpjpe
    parser_mpjpe.add_argument('--hidden_dim', default=128, type=int, required=False)  
    parser_mpjpe.add_argument('--num_blocks', default=10, type=int, required=False)  
    parser_mpjpe.add_argument('--tokens_mlp_dim', default=20, type=int, required=False)
    parser_mpjpe.add_argument('--channels_mlp_dim', default=128, type=int, required=False)  
    parser_mpjpe.add_argument('--regularization', default=0.1, type=float, required=False) 
    parser_mpjpe.add_argument('--pose_dim', default=54, type=int, required=False)
    parser_mpjpe.add_argument('--delta_x', type=bool, default=True, help='predicting the difference between 2 frames')
    parser_mpjpe.add_argument('--lr', default=0.001, type=float, required=False)  
    args = parser_mpjpe.parse_args()
    


    print(args)

    model = MlpMixer(num_classes=args.pose_dim, num_blocks=args.num_blocks,
                     hidden_dim=args.hidden_dim, tokens_mlp_dim=args.tokens_mlp_dim,
                     channels_mlp_dim=args.channels_mlp_dim, seq_len=args.input_n,
                     pred_len=args.output_n, activation=args.activation,
                     mlp_block_type='normal', regularization=args.regularization,
                     input_size=args.pose_dim, initialization='none', r_se=args.r_se,
                     use_max_pooling=False, use_se=True)

    model = model.to(args.dev)


    model.load_state_dict(torch.load(args.model_path))
    
    
    model.eval ()
    
   
    test_mixer(model, args)
   
   

   
   
   
   
   


   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    
    
    