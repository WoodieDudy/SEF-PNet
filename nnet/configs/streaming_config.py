import os
import random
import string


gpus = "0"
epochs = 70
checkpoint = os.getenv('EXP_NAME', 'exp_test' + ''.join(random.choices(string.digits, k=3)))
resume = None
batch_size = 128
num_workers = 8


fs = 8000 
chunk_len = 4  # (s)
chunk_size = chunk_len * fs 

nnet_conf = {
    "win_len": 512, 
    "win_inc": 128, 
    "fft_len": 512, 
    "win_type": "sqrthann",
    "kernel_size": (3, 3),
    "stride1": (1, 1), 
    "stride2": (1, 2), 
    "paddings": (2, 0),
    "output_padding": (0, 0),
    "tcn_dims": 384, 
    "tcn_blocks": 10,
    "tcn_layers": 2,
    "causal": True,
    "num_spks": 1 
}


# data configure:
train_dir = "data/train/"
dev_dir = "data/dev/"

train_data = {
    "mix_scp": train_dir + "mix_clean.scp", 
    "ref_scp": train_dir + "ref.scp",
	"aux_scp": train_dir + "auxs1.scp",
    "sample_rate": fs,
    # "add_rir": False
}

dev_data = { 
    "mix_scp": dev_dir + "mix_clean.scp", 
    "ref_scp": dev_dir + "ref.scp",
	"aux_scp": dev_dir + "auxs1.scp",
    "sample_rate": fs,
    # "add_rir": False
}

# trainer config
adam_kwargs = {
    "lr": 0.5e-3, 
    "weight_decay": 1e-5, 
}

trainer_conf = {
    "optimizer": "adamw", 
    "optimizer_kwargs": adam_kwargs, 
    "min_lr": 1e-8, 
    "patience": 25, 
    "factor": 0.5, 
    "metric_change_threshold": 0.1,
    "logging_period": 10  
}
