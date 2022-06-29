# DATA
dataset='Tusimple'
data_root = 'D:/TuSimple/test'

# TRAIN
epoch = 100
batch_size = 1
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 8e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '34'
griding_num = 200
use_aux = False

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = 'D:/log'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = 'ep099-34high-96.61.pth'
test_work_dir = 'D:/testdir'

num_lanes = 7