import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

def sample_x0(bs):
    return torch.randn((bs, 2))

def sample_x1(bs):
    x, _ = make_swiss_roll(n_samples=bs, noise=0.5)
    # Make two-dimensional to easen visualization
    x = x[:, [0, 2]]

    x = (x - x.mean()) / x.std()
    return x.astype('float32')
    # p = torch.rand((bs,))
    # return torch.tensor([[.5, .5]]) * (p < 0.25)[:, None] + \
    #         torch.tensor([[-.5, .5]]) * ((p >= 0.25) & (p < 0.5))[:, None] + \
    #         torch.tensor([[.5, -.5]]) * ((p >= 0.5) & (p < 0.75))[:, None] + \
    #         torch.tensor([[-.5, -.5]]) * (p >= 0.75)[:, None] + \
    #         torch.randn((bs, 2)) * 0.1

def get_time_embedding(time):
    freqs = torch.pow(10000, -torch.arange(start=0, end=16, dtype=torch.float32) / 16)
    x = (time.to(torch.float32))[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

class Block(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.norm_feat = nn.LayerNorm(ch)
        self.norm_merged = nn.LayerNorm(ch)
        self.feat_proj = nn.Linear(ch, ch)
        self.time_proj = nn.Linear(ch, ch)
        self.merged_proj = nn.Linear(ch, ch)

    def forward(self, x, time):
        residual = x

        # bypass
        merged = self.feat_proj(F.silu(self.norm_feat(x))) + self.time_proj(F.silu(time))
        merged = self.merged_proj(F.silu(self.norm_merged(merged)))

        return merged + residual


class Net(nn.Module):
    def __init__(self, ch=32, n_blocks=4):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(32, ch),
            nn.SiLU(),
            nn.Linear(ch, ch)
        )
        self.in_proj = nn.Linear(2, ch)
        self.output_layer = nn.Sequential(
            nn.LayerNorm(ch),
            nn.SiLU(),
            nn.Linear(ch, 2)
        )
        self.blocks = nn.Sequential(*[Block() for _ in range(n_blocks)])

    def forward(self, x, time):
        time = self.time_emb(time)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, time)
        return self.output_layer(x)

def train(model: nn.Module, optimizer: torch.optim.Optimizer, device='cpu', bs=2048, T=1000, beta_min=1e-4, beta_max=0.02, max_iters=10_000, save_every=20):
    betas = torch.linspace(beta_min, beta_max, T)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, 0)
    model.to(device)
    model.train()
    for i in range(max_iters):
        x1 = sample_x1(bs)
        t = torch.randint(1, T+1, size=(bs,))
        noise = sample_x0(bs)
        model_input = torch.sqrt(alpha_bars[t-1])[:, None] * x1 + torch.sqrt(1 - alpha_bars[t-1])[:, None] * noise
        t_emd = get_time_embedding(t)
        model_output = model(model_input.to(device), t_emd.to(device))
        loss = ((noise.to(device) - model_output) ** 2).sum() / bs
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'iter: {i}, loss: {loss.item()}')

        if i % save_every == 0:
            samples = sample(model, bs=1000, device=device)
            plt.figure()
            plt.scatter(samples[:, 0], samples[:, 1])
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.savefig(f'ddpm_vis/{i}.png')
            plt.close()
            model.train()

def sample(model: nn.Module, device='cpu', bs=64, T=1000, beta_min=1e-4, beta_max=0.02):
    betas = torch.linspace(beta_min, beta_max, T)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, 0)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = sample_x0(bs).to(device)
        for t in range(T, 0, -1):
            if t > 1:
                z = sample_x0(bs).to(device)
            else:
                z = torch.zeros_like(x)
            t_emb = get_time_embedding(torch.full((bs,), t)).to(device)
            x = 1 / (alphas[t-1] ** 0.5) * (x - (1 - alphas[t-1]) / ((1 - alpha_bars[t-1]) ** 0.5) * model(x, t_emb)) + (betas[t-1] ** 0.5) * z
    return x.cpu().numpy()

if __name__ == "__main__":
    device = 'mps'
    model = Net()
    opti = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train(model, opti, device, max_iters=1000)
    samples = sample(model, bs=1000)
    
    plt.figure()
    x0s = sample_x0(1000)
    plt.scatter(x0s[:, 0], x0s[:, 1])
    
    x1s = sample_x1(1000)
    plt.scatter(x1s[:, 0], x1s[:, 1], c='orange')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()