import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Unet
from diffusion import Diffusion
from utils import create_dir

def create_model(opts):
    """builds the generators."""
    U = Unet(dim=opts.conv_dim, channels=3, dim_mults=(1, 2, 4,))
    if torch.cuda.is_available():
        U.cuda()
        print("Models moved to GPU.")
    return U

def sample(model, diffusion, image_size, batch_size=16, channels=3, seed=None):
    if seed is None:
        seed = torch.seed()  
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    return diffusion.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

def main(opts):
    # create checkpoint and sample directories
    create_dir(opts.sample_dir)

    # initialize diffusion
    diffusion = Diffusion(timesteps=opts.denoising_steps)

    # create model
    U = create_model(opts)
    
    # load checkpoint
    if opts.checkpoint_file:
        checkpoint_path = opts.checkpoint_file
    else:
        checkpoint_path = os.path.join(opts.checkpoint_dir, 'diffusion.pth')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(checkpoint_path):
        U.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return
    U.to(device)
    U.eval()

    # generate samples in batches until reaching the requested count
    print(f"Generating {opts.num_samples} samples...")
    seed = getattr(opts, "seed", None)
    total_generated = 0
    batch_index = 0

    while total_generated < opts.num_samples:
        current_batch_size = min(opts.batch_size, opts.num_samples - total_generated)
        batch_seed = seed + batch_index if seed is not None else None

        samples = sample(
            U,
            diffusion,
            opts.image_size,
            batch_size=current_batch_size,
            channels=3,
            seed=batch_seed,
        )

        final_samples = samples[-1]  # get the final denoised images

        for i in range(current_batch_size):
            img = final_samples[i]
            img = np.clip((img + 1.0) * 0.5, 0, 1)
            img = np.transpose(img, (1, 2, 0))

            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            save_path = os.path.join(opts.sample_dir, f"sample_{total_generated + i}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved sample to {save_path}")

        total_generated += current_batch_size
        batch_index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--conv_dim", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16, help="number of samples generated per diffusion batch")
    parser.add_argument("--denoising_steps", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="path to specific checkpoint file (overrides checkpoint_dir)")
    parser.add_argument("--seed", type=int, default=None, help="optional seed; omit for fresh randomness each run")
    parser.add_argument("--sample_dir", type=str, default="./samples")
    
    args = parser.parse_args()
    main(args)
