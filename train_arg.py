from models.dual_inception import DualInception
from models.off_apexnet import OffApexNet
from models.ststnet import STSTNet
from models.macnn import MACNN
from models.micro_attention import MicroAttention
from models.off_tanet import OffTANet

from torch import nn,optim
import torch
from data import CASME2

model_names = ['dual-inception','attention']
model_name = model_names[-1]

from_file = 1
data_filename = ==========
data_class = =============
data_args = {'path':'dataset/CASME2','img_sz' : 112,'calculate_strain' : True,'raw_img' : False}

n_epochs = 201
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
observed_epochs = set([i for i in range(0,n_epochs,20)])

batch_size = 64
model_class = OffTANet
model_args = {'net_type' : 'ta'}

optimizer_class = optim.Adam
optimizer_args = {'lr' : ===}

scheduler_class = optim.lr_scheduler.StepLR
scheduler_args = {'step_size' : ===,'gamma' : ===}

print_debug_info = 1
train_process_filename = 'off-tanet.pkl'

if model_name == 'dual-inception':
    data_filename = =============
    data_args['img_sz'] = =======
    data_args['calculate_strain'] = False
    model_class = DualInception
    n_epochs = ==================
    observed_epochs = set([i for i in range(0,n_epochs,20)])
    optimizer_args = {'lr' : ===}
    scheduler_args = {'step_size' : ===,'gamma' : ===}
    train_process_filename = 'dual-inception-result.pkl'
elif model_name == 'attention':
    data_filename = ============
    data_args['img_sz'] = ======
    data_args['calculate_strain'] = True
    model_class = Attention
    model_args = {==============}
    n_epochs = =================
    observed_epochs = set([i for i in range(0,n_epochs,10)])
    optimizer_class = optim.SGD
    optimizer_args = {'lr': ===,'weight_decay': ===,'momentum': ===}
    scheduler_args = {'step_size' : ===,'gamma' : ===}
    train_process_filename = 'attention-result.pkl'
    batch_size = ===============
