# config for training GPT-2 (124M) on a single L40S 48GB
# $ python train.py config/train_gpt2_l40s.py

wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-124M-l40s'

# 32 batch size * 1024 block size * 15 gradaccum = 491,520 tokens per iter, same as train_gpt2.py
# micro batch 32 measured fastest of 12/24/32/48 on L40S
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 15

# ~9.8B tokens, about one epoch of OpenWebText, ~19h at ~3.45s/iter
max_iters = 20000
lr_decay_iters = 20000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# L40S dense bfloat16 tensor peak is 362 TFLOPS
flops_promised = 362e12
