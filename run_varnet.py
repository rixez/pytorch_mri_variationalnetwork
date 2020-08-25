from models import VariationalNetwork
import torch
import argparse
import pytorch_lightning as pl
from data_utils import *
from torch.utils.data import DataLoader
from pathlib import Path
from torchsummary import summary
from misc_utils import print_options
import os
parser = argparse.ArgumentParser(description='Variational network arguments')

# Data IO
parser.add_argument('--name',             type=str, default='coronal_pd_fs',     help='name of the dataset to use')
parser.add_argument('--root_dir',         type=str, default='data/knee',         help='directory of the data')
parser.add_argument('--sampling_pattern', type=str, default='cartesian_with_os', help='type of sampling pattern')

# Network configuration
parser.add_argument('--features_out',     type=int, default=24,    help='number of filter for convolutional kernel')
parser.add_argument('--num_act_weights',  type=int, default=31,    help='number of RBF kernel for activation function')
parser.add_argument('--num_stages',       type=int, default=10,    help='number of stages in the network')
parser.add_argument('--activation',       type=str, default='rbf', help='activation function to use (rbf or relu)')
# Training and Testing Configuration
parser.add_argument('--mode',             type=str,   default='train',               help='train or eval')
parser.add_argument('--optimizer',        type=str,   default='adam',                help='type of optimizer to use for training')
parser.add_argument('--loss_type',        type=str,   default='complex',             help='compute loss on complex or magnitude image')
parser.add_argument('--lr',               type=float, default=1e-4,                  help='learning rate')
parser.add_argument('--epoch',            type=int,   default=100,                   help='number of training epoch')
parser.add_argument('--batch_size',       type=int,   default=1,                     help='batch size')
parser.add_argument('--gpus',             type=str,   default='0',        	         help='gpu id to use')
parser.add_argument('--save_dir',         type=str,   default='exp/basic_varnet',    help='directory of the experiment')
parser.add_argument('--momentum',         type=float, default=0.,                    help='momentum for the optimizer')
parser.add_argument('--loss_weight',      type=float, default=1.,                    help='weight for the loss function')
parser.add_argument('--error_scale',      type=float, default=1.,                    help='how much to magnify the error map for display purpose')

args = parser.parse_args()
print_options(parser,args)
args = vars(args)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']
# setting up network
varnet = VariationalNetwork(**args)

# setting up data loader
dataset = KneeDataset(**args)
if args['mode'] == 'train':
	shuffle = True
else:
	shuffle = False
dataloader = DataLoader(dataset,batch_size=args['batch_size'],shuffle=shuffle,num_workers=8,pin_memory=True)

# start training
save_dir = Path(args['save_dir'])
save_dir.mkdir(parents=True,exist_ok=True)
trainer = pl.Trainer(gpus=len(args['gpus'].split(',')),max_epochs=args['epoch'])

if args['mode'] == 'train':
	trainer.fit(varnet,dataloader)
	torch.save(varnet.state_dict(), str(save_dir/'varnet.h5'))
elif args['mode'] == 'eval':
	varnet.load_state_dict(torch.load(str(save_dir/'varnet.h5')))
	trainer.test(varnet,test_dataloaders=dataloader)


