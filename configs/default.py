from datasets.transform import RAWTrainTransform, RAWEvalTransform

# Training parameters
seed = 1
output_dir = "experiments"
device = "cuda"
batch_size = 128
num_workers = 8
lr = 0.0001
aux_lr = 0.001
# blr = 0.0001
# aux_blr = 0.001
num_epochs = 200
lr_reduce_patience = 10
lr_reduce_factor = 0.1
multistep = False
milestones = [350, 390, 430, 470, 510, 550, 590]
gamma = 0.9
clip_grad = 1.0
prefetch_factor = 4
pin_memory = True

warmup = False
warmup_epochs = 5


patch_sz = 64  # B * 3 * 64 * 64
transform_train = RAWTrainTransform(patch_sz)
transform_val = RAWEvalTransform(patch_sz)

# Dist Params
dist_backend = "nccl"
dist_url = "env://"
world_size = 1
