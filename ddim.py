import torch
import numpy as np

class DDIMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, sample_steps=10, is_inversion=True, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.zero = torch.tensor(0.0)

        self.generator = generator
        self.sample_steps = sample_steps
        self.is_inversion = is_inversion
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50, sample_steps=10, is_inversion=False):
        self.sample_steps = sample_steps
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        num_inference_steps = num_inference_steps // sample_steps
        self.is_inversion = is_inversion
        if not is_inversion:
            timesteps = (np.arange(0, num_inference_steps) * (step_ratio * sample_steps)).round()[::-1].copy().astype(np.int64)
            timesteps_next = (np.arange(0, num_inference_steps + 1) * (step_ratio * sample_steps)).round()[::-1].copy().astype(np.int64)
        else:
            timesteps = (np.arange(0, num_inference_steps) * (step_ratio * sample_steps)).round().copy().astype(np.int64)
            timesteps_next = (np.arange(0, num_inference_steps + 1) * (step_ratio * sample_steps)).round().copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        self.timesteps_next = torch.from_numpy(timesteps_next)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps * self.sample_steps
        return prev_t
    
    def _get_next_timestep(self, timestep: int) -> int:
        next_t = timestep + self.num_train_timesteps // self.num_inference_steps * self.sample_steps
        return next_t
    
    def set_strength(self, strength=1):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor, is_inversion=False):
        t = timestep
        if not self.is_inversion:
            prev_t = self._get_previous_timestep(t)

            # 1. compute alphas, betas
            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev


            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from ddim paper
            pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (12) from ddim paper
            pred_original_sample_coeff = alpha_prod_t_prev ** 0.5
            pred_noise_coeff = beta_prod_t_prev ** 0.5

            pred_sample_direction = pred_noise_coeff * model_output
            # 5. Compute predicted previous sample
            # See formula (12) from ddim paper
            pred_prev_sample = pred_original_sample_coeff * pred_original_sample + pred_sample_direction

            return pred_prev_sample
        
        else:
            next_t = self._get_next_timestep(t)

            # 1. compute alphas, betas
            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_next = self.alphas_cumprod[next_t] if next_t < self.num_train_timesteps else self.zero
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_next = 1 - alpha_prod_t_next


            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from ddim paper
            pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (12) from ddim paper
            pred_original_sample_coeff = alpha_prod_t_next ** 0.5
            pred_noise_coeff = beta_prod_t_next ** 0.5

            next_sample_direction = pred_noise_coeff * model_output
            # 5. Compute predicted previous sample
            # See formula (12) from ddim paper
            pred_next_sample = pred_original_sample_coeff * pred_original_sample + next_sample_direction

            return pred_next_sample

    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

        

    