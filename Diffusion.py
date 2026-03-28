import torch
from tqdm import tqdm
import torch.nn.functional as F

class Diffusion:
    
    def __init__(self, steps, start=1e-4, end=2e-2, beta_type="linear", device='cpu', **kwargs):
        
        assert beta_type in ['linear', 'cosine', 'sigmoid', 'quadratic']
        
        self.steps      = steps
        self.beta_start = start
        self.beta_end   = end
        self.device     = device
        
        if beta_type in ['cosine']:
            betas = self.cosine_beta_schedule(kwargs['s'])
        elif beta_type in ['linear']:
            betas = self.linear_beta_schedule()
        elif beta_type in ['quadratic']:
            betas = self.quadratic_beta_schedule()
        elif beta_type in ['sigmoid']:
            betas = self.sigmoid_beta_schedule()
        
        betas = betas.to(device)
        alphas = 1 - betas
        
        alphas_bar          = alphas.cumprod(0)
        alphas_bar_pre      = F.pad(alphas_bar[:-1], (1, 0), value=1.0)
        
        # 正向传播 p, 反向传播 q
        # 影响均值用alpha，影响标准差用beta，影响预测用gamma
        
        self.p_alphas = alphas_bar.sqrt()
        self.p_betas  = (1 - alphas_bar).sqrt()
        
        self.q_alphas = 1 / alphas.sqrt()
        self.q_betas  = betas * (1 - alphas_bar_pre) / (1 - alphas_bar)
        self.q_gammas = betas / (1 - alphas_bar).sqrt()
    
    def do_noise(self, x, t):
        
        t = t.to(torch.long)
        alpha = self.p_alphas[t, None, None, None]
        beta  = self.p_betas[t,  None, None, None]
        noise = torch.randn_like(x)
        
        return alpha * x + beta * noise, noise
    
    def do_sample(self, model, x, t):
        
        t = t.to(torch.long)
        alpha = self.q_alphas[t, None, None, None]
        beta  = self.q_betas[t,  None, None, None]
        gamma = self.q_gammas[t, None, None, None]
        
        pred_noise = model(x, t)
        
        mean  = alpha * (x - gamma * pred_noise)
        pred  = mean + beta * torch.randn_like(x)
        
        return pred
    
    def sample_all(self, model, n,  img_size):
        
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, img_size, img_size)).to(self.device)
            t = torch.ones([n]).to(torch.long).to(self.device)
            for i in tqdm(list(reversed(range(1, self.steps))), desc="sampling"):
                ti = t * i
                x = self.do_sample(model, x, ti)
        model.train()
                
        min_ = x.flatten(1, -1).min(dim=-1)[0][:, None, None, None]
        x    = x - min_
        max_ = x.flatten(1, -1).max(dim=-1)[0][:, None, None, None]
        x    = x / max_
        
        return x * 255
        

    def cosine_beta_schedule(self, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = self.steps + 1
        x = torch.linspace(0, self.steps, steps)
        alphas_cumprod = torch.cos(((x / self.steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.steps)

    def quadratic_beta_schedule(self):
        return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.steps) ** 2

    def sigmoid_beta_schedule(self):
        betas = torch.linspace(-6, 6, self.steps)
        return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start


