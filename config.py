# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -5.0 # -4.5
lr = 10 ** log10_lr
epochs = 500
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 1
lamda_guide = 10
lamda_low_frequency = 1
device_ids = [3]

# Train:
batch_size = 16
cropsize = 128
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 128
batchsize_val = 16
shuffle_val = False
val_freq = 10


# Dataset
TRAIN_PATH = '/home/zzc/deeplearn_test/DRHiNet/data/train/train_class/'
VAL_PATH = '/home/zzc/deeplearn_test/DRHiNet/data/val/val_class/'
format_train = 'jpg'
format_val = 'jpg'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = '/home/zzc/deeplearn_test/DRHiNet/model/model_checkpoint_00400'
MODEL_SAVE = '/home/zzc/deeplearn_test/DRHiNet/model/'
checkpoint_on_error = True
SAVE_freq = 20

IMAGE_PATH = '/home/zzc/deeplearn_test/DRHiNet/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'
IMAGE_PATH_attack = IMAGE_PATH + 'attack/'
# Load:
suffix = '.pt'
tain_next = False
trained_epoch = 0

# Secret message
messages_length = 128
messages_length = 128

# Attack
attack = True