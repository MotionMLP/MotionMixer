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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



def get_log_dir(out_dir):
    dirs = [x[0] for x in os.walk(out_dir)]
    if len(dirs ) < 2:
        log_dir = os.path.join(out_dir, 'exp0')
        os.mkdir(log_dir)
    else:
        log_dir = os.path.join(out_dir, 'exp%i'%(len(dirs)-1))
        os.mkdir(log_dir)

    return log_dir


#%%
def train(model, model_name, args):
    
    joint_used=np.arange(4,22)

    log_dir = get_log_dir(args.root)
    tb_writer = SummaryWriter(log_dir=log_dir)
    print('Save data of the run in: %s'%log_dir)

    device = args.dev

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)

    train_loss, val_loss, test_loss = [], [], []


    dataset = Datasets(args.data_dir, args.input_n,
                    args.output_n, args.skip_rate, split=0)
    
    vald_dataset = Datasets(args.data_dir, args.input_n,
                        args.output_n, args.skip_rate, split=1)
    


    
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_worker, pin_memory=True)
    vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_worker, pin_memory=True)

    
    for epoch in range(args.n_epochs):
        print('Run epoch: %i'%epoch)
        running_loss = 0
        n = 0
        model.train()
        for cnt, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch = batch.to(device)
            batch = batch[:, :, joint_used]
            batch_dim = batch.shape[0]
            n += batch_dim

            
            sequences_train = batch[:,0:args.input_n,:,:].reshape(
                -1, args.input_n, args.pose_dim)
            sequences_gt = batch[:,args.input_n:args.input_n+args.output_n,:,:].reshape(-1, args.output_n, args.pose_dim)


            optimizer.zero_grad()

            sequences_predict=model(sequences_train)
            
            loss=mpjpe_error(sequences_predict,sequences_gt)*1000
            
            if cnt % 200 == 0:
                 print('[%d, %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))    

            loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad)

            optimizer.step()

            running_loss += loss*batch_dim

        train_loss.append(running_loss.detach().cpu()/n)
        model.eval()
        with torch.no_grad():
            running_loss = 0
            n = 0
            for cnt, batch in enumerate(vald_loader):
                batch = batch.to(device)
                batch = batch[:, :, joint_used]
                batch_dim = batch.shape[0]
                n += batch_dim

               
                sequences_train = batch[:,0:args.input_n,:,:].reshape(
                    -1, args.input_n, args.pose_dim)
                sequences_gt = batch[:,args.input_n:args.input_n+args.output_n,:,:].reshape(-1, args.output_n, args.pose_dim)
            

                sequences_predict=model(sequences_train)
            
                loss=mpjpe_error(sequences_predict,sequences_gt)*1000
                
                if cnt % 200 == 0:
                    print('[%d, %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))    


                running_loss += loss*batch_dim
            val_loss.append(running_loss.detach().cpu()/n)
        if args.use_scheduler:
            scheduler.step()

       
        test_loss.append(test_mpjpe(model, args))
        

        tb_writer.add_scalar('loss/train', train_loss[-1].item(), epoch)
        tb_writer.add_scalar('loss/val', val_loss[-1].item(), epoch)

        torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))
        # TODO write something to save the best model
        if (epoch+1)%1==0:
            print('----saving model-----')
            torch.save(model.state_dict(),os.path.join(args.model_path,model_name))



#%%
def test_mpjpe(model, args):

    device = args.dev
    model.eval()
    accum_loss = 0
    n_batches = 0  # number of batches for all the sequences


    running_loss = 0
    n = 0
       

    Dataset = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=2)
    loader_test = DataLoader( Dataset,
                                 batch_size=args.batch_size,
                                 shuffle =False,
                                 num_workers=0)
        
       
                
                
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

#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False) # Parameters for mpjpe
    parser.add_argument('--data_dir', type=str, default='../data_amass/', help='path to the unziped dataset directories(H36m/AMASS/3DPW)')  
    parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
    parser.add_argument('--output_n', type=int, default=25, help="number of model's output frames")
    parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5], help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')
    parser.add_argument('--root', default='./runs', type=str, help='root path for the logging') #'./runs'

    parser.add_argument('--activation', default='gelu', type=str, required=False)  
    parser.add_argument('--r_se', default=8, type=int, required=False)

    parser.add_argument('--n_epochs', default=50, type=int, required=False)
    parser.add_argument('--batch_size', default=200, type=int, required=False)  # 100  50  in all original 50
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
    parser.add_argument('--actions_to_consider', default='all', help='Actions to visualize.Choose either all or a list of actions')
    parser.add_argument('--batch_size_test', type=int, default=256, help='batch size for the test set')
    parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'], help='choose data split to visualize from(train-val-test)')
    parser.add_argument('--loss_type', type=str, default='mpjpe', choices=['mpjpe', 'angle'])

    args = parser.parse_args()

    parser_mpjpe = argparse.ArgumentParser(parents=[parser]) # Parameters for mpjpe
    parser_mpjpe.add_argument('--hidden_dim', default=128, type=int, required=False)  
    parser_mpjpe.add_argument('--num_blocks', default=5, type=int, required=False) 
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

    print('total number of parameters of the network is: ' +
          str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model_name = 'h36_3d_'+str(args.output_n)+'frames_ckpt'

#%%
    train(model, model_name, args)
    test_mpjpe(model, args)



