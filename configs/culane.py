# DATA
dataset='CULane'
data_root = 'D:/Ultra-Fast-Lane-Detection-master'

# TRAIN
epoch = 50
batch_size = 32
optimizer = 'Adam'  #['SGD','Adam']
learning_rate = 4e-5
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = False
griding_num = 200
backbone = '34'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/home/log'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = 'ep027-71.07.pth'
test_work_dir = '/home/testlog'

num_lanes = 4




