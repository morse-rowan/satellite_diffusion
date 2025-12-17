import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from model import Unet
from diffusion import DDPM, DDIM
from utils import create_dir
from tqdm.auto import tqdm

def create_gif(samples_history, save_path, fps=20):
    frames = []
    timesteps = len(samples_history)
    
    if np.allclose(samples_history[0], samples_history[-1], atol=1e-4):
        print("Warning: GIF frames are identical. Diffusion might not be working.")

    for t_idx, img_data in enumerate(samples_history):
        img_data = np.clip((img_data + 1.0) * 0.5, 0, 1)
        img_data = np.transpose(img_data, (1, 2, 0)) # H, W, C
        img_uint8 = (img_data * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img_uint8)
        
        #draw = ImageDraw.Draw(pil_img)
        #current_ts = timesteps - 1 - t_idx
                
        frames.append(pil_img)
        
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000/fps,
        loop=0
    )

def create_model(opts):
    """builds the generators."""
    U = Unet(dim=opts.conv_dim, channels=3, dim_mults=(1, 2, 4,8), image_size=opts.image_size)
    if torch.cuda.is_available():
        U.cuda()
        print("Models moved to GPU.")
    return U

def ddim_sample_loop(model, diffusion, shape, sampling_steps, eta=0.0):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    # Use the training schedule (e.g. 500 steps)
    total_steps = diffusion.timesteps
    
    # evenly spaced subsequence of steps (e.g. [490, 480, ..., 0])
    times = list(range(0, total_steps, total_steps // sampling_steps))
    times = sorted(times, reverse=True)
    
    # add the final step -1 (which represents t=0 fully denoised) for the loop logic
    #  iterate through pairs (t, prev_t)
    
    with torch.no_grad():
        for i, t in enumerate(tqdm(times, desc="DDIM Sampling")):
            # determine the previous time step (the one we are going TO)
            prev_t = times[i+1] if i < len(times) - 1 else -1
            
            # 1. predict noise
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            pred_noise = model(img, t_tensor)
            
            # 2. get alphas from the schedule
            alpha_t = diffusion.alphas_cumprod[t].to(device)
            alpha_t_prev = diffusion.alphas_cumprod[prev_t].to(device) if prev_t >= 0 else torch.tensor(1.0, device=device)
            
            # reshape for broadcasting
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            alpha_t_prev = alpha_t_prev.view(-1, 1, 1, 1)
            
            # 3. compute predicted x0 (denoised image)
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            
            # 4. compute direction pointing to x_t
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * pred_noise
            
            # 5. compute x_{t-1}
            noise = sigma_t * torch.randn_like(img)
            img = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + noise
            
            # Store on CPU to avoid GPU memory accumulation
            imgs.append(img.detach().cpu().numpy())
            
            # Clear cache periodically to prevent fragmentation
            if (i + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    return imgs

def sample(model, diffusion, image_size, batch_size=16, channels=3, seed=None, sampling_steps=None, method='ddpm', eta=0.0):
    if seed is None:
        seed = torch.seed()  
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    if method == 'ddim':
        return ddim_sample_loop(model, diffusion, (batch_size, channels, image_size, image_size), sampling_steps, eta)
    else:
        return diffusion.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

def main(opts):
    # create checkpoint and sample directories
    create_dir(opts.sample_dir)

    # initialize diffusion
    if opts.sampling_method == 'ddim':
        diffusion = DDIM(timesteps=opts.train_timesteps, eta=opts.eta)
    else:
        diffusion = DDPM(timesteps=opts.train_timesteps)
    
    # move diffusion to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion.to(device)

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
            sampling_steps=opts.denoising_steps,
            method=opts.sampling_method,
            eta=opts.eta
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

            if opts.save_gif:
                sample_history = [s[i] for s in samples]
                gif_path = os.path.join(opts.sample_dir, f"sample_{total_generated + i}.gif")
                create_gif(sample_history, gif_path)
                print(f"Saved gif to {gif_path}")

        total_generated += current_batch_size
        batch_index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--conv_dim", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16, help="number of samples generated per diffusion batch")
    parser.add_argument("--train_timesteps", type=int, default=500, help="number of timesteps the model was trained on")
    parser.add_argument("--denoising_steps", type=int, default=50, help="number of steps to use for sampling (inference)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="path to specific checkpoint file (overrides checkpoint_dir)")
    parser.add_argument("--seed", type=int, default=None, help="optional seed; omit for fresh randomness each run")
    parser.add_argument("--sample_dir", type=str, default="./samples")
    parser.add_argument("--sampling_method", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--eta", type=float, default=0.0, help="eta for ddim sampling")
    parser.add_argument("--save_gif", action="store_true")
    
    args = parser.parse_args()
    main(args)
