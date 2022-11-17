from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()


config.TRAIN.batch_size = 8 
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9


config.TRAIN.n_epoch_init = 75

config.TRAIN.n_epoch = 56
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

print("config.TRAIN.decay_every:", config.TRAIN.decay_every)

config.TRAIN.hr_img_path = 'train_data_out' 
config.TRAIN.lr_img_path = 'train_data_in' 


config.VALID = edict()


config.VALID.hr_img_path = 'test_data_out' 
config.VALID.lr_img_path = 'test_data_in' 

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
