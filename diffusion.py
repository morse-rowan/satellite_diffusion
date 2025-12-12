import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import random

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DDPM:
    def __init__(self, timesteps=500, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    # forward diffusion 
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)

        x_self_cond = None
        if denoise_model.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = denoise_model(x_noisy, t)
                x_self_cond.detach_()

        predicted_noise = denoise_model(x_noisy, t, x_self_cond)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        device = next(model.parameters()).device
        b = shape[0]
        # start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(
            reversed(range(0, self.timesteps)), desc="sampling loop time step", total=self.timesteps
        ):
            img = self.p_sample(
                model, img, torch.full((b,), i, device=device, dtype=torch.long), i
            )
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

class DDIM(DDPM):
    def __init__(self, timesteps=500, beta_start=0.0001, beta_end=0.02, eta=0.0):
        super().__init__(timesteps, beta_start, beta_end)
        self.eta = eta

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # extract values
        alpha_t = extract(self.alphas_cumprod, t, x.shape)
        alpha_t_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_alpha_t = extract(self.sqrt_alphas_cumprod, t, x.shape)

        # predict noise
        pred_noise = model(x, t)

        # predict x0
        pred_x0 = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

        # calculate direction pointing to xt
        sigma_t = self.eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        dir_xt = torch.sqrt(1.0 - alpha_t_prev - sigma_t**2) * pred_noise

        # random noise
        noise = sigma_t * torch.randn_like(x)

        # calculate x_{t-1}
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + noise
        return x_prev

