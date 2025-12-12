import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Unet
from diffusion import DDPM
from dataset import get_data_loader
from utils import create_dir, to_var, EMA
from torch.amp import autocast, GradScaler
import copy

def create_model(opts):
    """builds the generators."""
    U = Unet(dim=opts.conv_dim, channels=3, dim_mults=(1, 2, 4,))
    if torch.cuda.is_available():
        U.cuda()
        print("Models moved to GPU.")
    return U

def training_loop(train_dataloader, opts, logger):
    """runs the training loop."""
    
    # Initialize Diffusion
    diffusion = DDPM(timesteps=opts.denoising_steps)

    # Create model
    U = create_model(opts)
    
    # Initialize EMA
    ema = EMA(0.995)
    ema_model = copy.deepcopy(U).eval().requires_grad_(False)

    # Create optimizer
    u_optimizer = optim.Adam(U.parameters(), opts.lr, [opts.beta1, opts.beta2])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize GradScaler for AMP
    scaler = GradScaler(device)
    global_step = 0

    for epoch in range(opts.num_epochs):
        epoch_start_time = time.time()
        data_load_time = 0.0
        gpu_time = 0.0
        iter_start = time.time()
        print(f"--- Epoch [{epoch}/{opts.num_epochs}] ---")

        for step, batch in enumerate(train_dataloader):
            data_load_time += time.time() - iter_start

            # Start GPU timer
            gpu_start = time.time()

            real_images = batch
            real_images = to_var(real_images)
            real_images = real_images.to(device)

            # 1. Sample t uniformally for every example in the batch
            t = torch.randint(
                low=0,
                high=opts.denoising_steps,
                size=(real_images.shape[0],),
                device=device,
            ).long()

            # 2. Get loss
            with autocast(device_type=device):
                loss = diffusion.p_losses(U, real_images, t)

            if step % opts.log_step == 0:
                print(f"Step {step} Loss: {loss.item()}")
                logger.add_scalar("Loss/train", loss.item(), global_step)

            u_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(u_optimizer)
            scaler.update()
            
            # Update EMA
            ema.step_ema(ema_model, U)

            gpu_time += time.time() - gpu_start
            iter_start = time.time()

            global_step += 1

        # Calculate epoch timing statistics
        epoch_total_time = time.time() - epoch_start_time
        data_percentage = (data_load_time / epoch_total_time) * 100 if epoch_total_time > 0 else 0
        gpu_percentage = (gpu_time / epoch_total_time) * 100 if epoch_total_time > 0 else 0
        print(f"Epoch {epoch} Timing --- Total: {epoch_total_time:.2f}s, Load: {data_load_time:.2f}s ({data_percentage:.1f}%), GPU: {gpu_time:.2f}s ({gpu_percentage:.1f}%)")

        if epoch % opts.checkpoint_every == 0:
            torch.save(U.state_dict(), os.path.join(opts.checkpoint_dir, f"diffusion_{epoch}.pth"))
            torch.save(ema_model.state_dict(), os.path.join(opts.checkpoint_dir, f"diffusion_ema_{epoch}.pth"))
            
    # Save final checkpoint
    torch.save(U.state_dict(), os.path.join(opts.checkpoint_dir, "diffusion.pth"))
    torch.save(ema_model.state_dict(), os.path.join(opts.checkpoint_dir, "diffusion_ema.pth"))

def main(opts):
    """loads the data, creates checkpoint and sample directories, and starts the training loop."""
    
    # Create logger
    if os.path.exists(opts.sample_dir):
        # Optional: clean up old logs
        pass
    logger = SummaryWriter(opts.sample_dir)

    # Create a dataloader for the training images
    dataloader = get_data_loader(opts.data_path, opts.metadata_path, opts)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)

    training_loop(dataloader, opts, logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument("--image_size", type=int, default=64, help="the side length n to convert images to nxn.")
    parser.add_argument("--conv_dim", type=int, default=32)
    parser.add_argument("--noise_size", type=int, default=100)

    # Training hyper-parameters
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--denoising_steps", type=int, default=500)

    # Data sources
    parser.add_argument("--data_path", type=str, required=True, help="path to image folder")
    parser.add_argument("--metadata_path", type=str, required=True, help="path to metadata json")
    parser.add_argument("--cloud_threshold", type=float, default=20.0, help="max cloud coverage %")
    parser.add_argument("--data_preprocess", type=str, default="vanilla")

    # Directories
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--sample_dir", type=str, default="./logs")
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=10)

    args = parser.parse_args()
    print(args)
    main(args)
