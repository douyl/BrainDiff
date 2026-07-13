import torch
from torch import nn, einsum
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)



class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        text_use_bert_cls=False,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,

    ):
        super().__init__()
        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32
        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, x, t, clip_denoised: bool, y=None, condition=None, res=None, hint=None, cond_scale=1.):
        if condition is None:
            x_recon = self.predict_start_from_noise(
                        x, t=t, noise=denoise_fn(x, t, y=y))  
        else:
            x_recon = self.predict_start_from_noise(
                        x, t=t, noise=denoise_fn(x, t, y=y, condition=condition))   

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, denoise_fn, x, t, y=None, condition=None, res=None, hint=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn,
            x=x, t=t, clip_denoised=clip_denoised, y=y, condition=condition, res=res, hint=hint, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    # @torch.inference_mode()
    # def sample(self, denoise_fn, z, y=None, res=None,cond_scale=1.,hint = None, strategy='ddpm', eta=0.0, ddim_steps= 100):
    #     if strategy == 'ddpm':
    #         return self.p_sample_loop(
    #                     denoise_fn, z, y, res, cond_scale, hint
    #                 )
    #     elif strategy == 'ddim':
    #         return self.p_sample_loop_ddim(
    #             denoise_fn, z, y, res, cond_scale, hint, eta, ddim_steps
    #         )
    #     else:
    #         raise NotImplementedError


    @torch.inference_mode()
    def p_sample_loop(self, denoise_fn, z, y=None, condition=None, res=None, cond_scale=1., hint = None, progress_bar=True):
        device = self.betas.device

        b = z.shape[0]
        img = default(z, lambda: torch.randn_like(z , device=device))

        indices = list(range(self.num_timesteps))[::-1]
        if progress_bar:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            img = self.p_sample(denoise_fn, img, torch.full(
                (b,), i, device=device, dtype=torch.long), y=y, condition=condition, res=res, cond_scale=cond_scale, hint=hint)

        return img


    @torch.inference_mode()
    def p_sample_loop_ddim(self, denoise_fn, z, y=None, condition=None, res=None, cond_scale=1., hint=None, eta=0.0, ddim_steps=100, progress_bar=True):
        device = self.betas.device
        b = z.shape[0]
        img = default(z, lambda: torch.randn_like(z, device=device))

        # DDIM time schedule
        times = torch.linspace(0, self.num_timesteps - 1, steps=ddim_steps).long().flip(0).to(device)

        indices = range(ddim_steps)
        if progress_bar:
            from tqdm.auto import tqdm
            indices = tqdm(range(ddim_steps))

        for i in indices:
            t = times[i]
            t_prev = times[i + 1] if i < ddim_steps - 1 else torch.tensor(0, device=device)
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((b,), t_prev, device=device, dtype=torch.long)

            img = self.p_sample_ddim(
                denoise_fn, img, t_batch, t_prev_batch,
                y=y, condition=condition, 
                res=res, hint=hint, cond_scale=cond_scale,
                eta=eta, clip_denoised=True
            )

        return img

    @torch.inference_mode()
    def p_sample_ddim(self, denoise_fn, x, t, t_prev, y=None, condition=None, res=None, hint=None, cond_scale=1., clip_denoised=True, eta=0.0):
        # Predict noise (ε_theta)
        if condition is None:
            noise_pred = denoise_fn(x, t, y=y)
        else:
            noise_pred = denoise_fn(x, t, y=y, condition=condition)

        # Predict x0
        x0 = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x0, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x0.ndim - 1)))
            x0 = x0.clamp(-s, s) / s

        # DDIM formula
        # alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        # alpha_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        alpha_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1, 1)
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        
        pred_dir = (1 - alpha_prev).sqrt() * noise_pred
        x_prev = alpha_prev.sqrt() * x0 + pred_dir

        if eta > 0:
            noise = torch.randn_like(x)
            x_prev = x_prev + sigma * noise

        return x_prev

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )


    def p_losses(self, denoise_fn, x_start, t, y=None, condition=None, res=None, noise=None, hint=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(y):
            y = bert_embed(
                tokenize(y), return_cls_repr=self.text_use_bert_cls)
            y = y.to(device)


        if condition is None:  # for base_model
            x_recon = denoise_fn(x_noisy, t, y=y)
        else:   # for controlnet
            x_recon = denoise_fn(x_noisy, t, y=y, condition=condition)

        # time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss
    
