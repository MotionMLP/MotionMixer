import torch
import os
from datasets.dataset_h36m import H36M_Dataset
from datasets.dataset_h36m_ang import H36M_Dataset_Angle
from utils.data_utils import define_actions
from torch.utils.data import DataLoader
from mlp_mixer import MlpMixer
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import argparse
from utils.utils_mixer import delta_2_gt, mpjpe_error, euler_error
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def test_pretrained(model,args):
 
  N = 0
  eval_frame = [1, 3, 7, 9, 13, 17, 21, 24]
  
  t_3d = np.zeros(len(eval_frame))
 
  t_3d_all = []
 
  model.eval()
  accum_loss=0  
  n_batches=0 # number of batches for all the sequences
  actions=define_actions(args.actions_to_consider)
  dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
  # joints at same loc
  joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
  index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
  joint_equal = np.array([13, 19, 22, 13, 27, 30])
  index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
 
  idx_eval = 7
 

  for action in actions:
    running_loss=0
    n=0
    dataset_test = H36M_Dataset(args.data_dir,args.input_n,args.output_n,args.skip_rate, split=2,actions=[action])

    test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
    for cnt,batch in enumerate(test_loader):
      with torch.no_grad():

        batch=batch.to(args.device)
        batch_dim=batch.shape[0]
        n+=batch_dim
       
       
        all_joints_seq=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]
        all_joints_seq_gt=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]

        sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used))

        sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used].view(-1,args.output_n,args.pose_dim)
       
        if  args.delta_x :            
            sequences_all = torch.cat((sequences_train, sequences_gt), 1)              
            sequences_all_delta =[sequences_all[:,1,:] - sequences_all[:,0,:]]
            for i in range(args.input_n+args.output_n-1):
                  sequences_all_delta.append(sequences_all[:,i+1,:] - sequences_all[:,i,:])
           
            sequences_all_delta = torch.stack((sequences_all_delta)).permute(1,0,2)
            sequences_train_delta =  sequences_all_delta[:, 0:args.input_n, :]
            sequences_predict=model(sequences_train_delta)
            sequences_predict = delta_2_gt (sequences_predict,sequences_train[:,-1,:])
            loss=mpjpe_error(sequences_predict,sequences_gt)
            
            
            
            sequences_gt_3d = sequences_gt.reshape (sequences_gt.shape[0],sequences_gt.shape[1],-1,3)
            sequences_predict_3d = sequences_predict.reshape (sequences_predict.shape[0],sequences_predict.shape[1],-1,3)
             
              #print (sequences_gt.shape)
             
            for k in np.arange(0, len(eval_frame)):
                  j = eval_frame[k]
                  t_3d[k] += torch.mean(torch.norm(sequences_gt_3d[:, j, :, :].contiguous().view(-1, 3) - sequences_predict_3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).item() * n
               
             
            N += n
                    
      
               
        else:
            sequences_predict=model(sequences_train)
            loss=mpjpe_error(sequences_predict,sequences_gt)  
           


        all_joints_seq[:,:,dim_used] = sequences_predict
        all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]
               
       
        all_joints_seq_gt[:,:,dim_used] = sequences_gt
        all_joints_seq_gt[:,:,index_to_ignore] = all_joints_seq_gt[:,:,index_to_equal]


        loss=mpjpe_error(all_joints_seq.view(-1,args.output_n,32,3),all_joints_seq_gt.view(-1,args.output_n,32,3))
        
        
        
        running_loss+=loss*batch_dim
        accum_loss+=loss*batch_dim

    print('loss at test subject for action : '+str(action)+ ' is: '+ str(running_loss/n))
    n_batches+=n
   
    t_3d_all.append (t_3d[idx_eval]/N)
   
  print('overall average loss in mm is: '+str(accum_loss/n_batches))
 
 

  print('overall final loss in mm is: ',np.mean(t_3d_all))
  
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False) # Parameters for mpjpe
    parser.add_argument('--data_dir', type=str, default='../data_h36m/', help='path to the unziped dataset directories(H36m/AMASS/3DPW)')  
    parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
    parser.add_argument('--output_n', type=int, default=25, help="number of model's output frames")
    parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5], help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')
    parser.add_argument('--root', default='./runs', type=str, help='root path for the logging') #'./runs'

    parser.add_argument('--activation', default='mish', type=str, required=False)  # 'mish', 'gelu'
    parser.add_argument('--r_se', default=8, type=int, required=False)

    parser.add_argument('--n_epochs', default=50, type=int, required=False)
    parser.add_argument('--batch_size', default=50, type=int, required=False)  # 100  50  in all original 50
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
    parser.add_argument('--model_path', type=str, default='./checkpoints/h36m_3d_25frames_ckpt', help='directory with the models checkpoints ')
    parser.add_argument('--actions_to_consider', default='all', help='Actions to visualize.Choose either all or a list of actions')
    parser.add_argument('--batch_size_test', type=int, default=256, help='batch size for the test set')
    parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'], help='choose data split to visualize from(train-val-test)')
    parser.add_argument('--loss_type', type=str, default='mpjpe', choices=['mpjpe', 'angle'])
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'])
         

    
    

    args = parser.parse_args()

    if args.loss_type == 'mpjpe':
        parser_mpjpe = argparse.ArgumentParser(parents=[parser]) # Parameters for mpjpe
        parser_mpjpe.add_argument('--hidden_dim', default=50, type=int, required=False)  
        parser_mpjpe.add_argument('--num_blocks', default=4, type=int, required=False)  
        parser_mpjpe.add_argument('--tokens_mlp_dim', default=20, type=int, required=False)
        parser_mpjpe.add_argument('--channels_mlp_dim', default=50, type=int, required=False)  
        parser_mpjpe.add_argument('--regularization', default=0.1, type=float, required=False)  
        parser_mpjpe.add_argument('--pose_dim', default=66, type=int, required=False)
        parser_mpjpe.add_argument('--delta_x', type=bool, default=True, help='predicting the difference between 2 frames')
        parser_mpjpe.add_argument('--lr', default=0.001, type=float, required=False)  
        args = parser_mpjpe.parse_args()
    
    elif args.loss_type == 'angle':
        parser_angle = argparse.ArgumentParser(parents=[parser]) # Parameters for angle
        parser_angle.add_argument('--hidden_dim', default=60, type=int, required=False) 
        parser_angle.add_argument('--num_blocks', default=3, type=int, required=False) 
        parser_angle.add_argument('--tokens_mlp_dim', default=40, type=int, required=False)
        parser_angle.add_argument('--channels_mlp_dim', default=60, type=int, required=False) 
        parser_angle.add_argument('--regularization', default=0.0, type=float, required=False)
        parser_angle.add_argument('--pose_dim', default=48, type=int, required=False)
        parser_angle.add_argument('--lr', default=1e-02, type=float, required=False) 
        args = parser_angle.parse_args()
    
    if args.loss_type == 'angle' and args.delta_x:
        raise ValueError('Delta_x and loss type angle cant be used together.')

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
    
    print('total number of parameters of the network is: ' +
          str(sum(p.numel() for p in model.parameters() if p.requires_grad)))



    
    test_pretrained(model, args)



