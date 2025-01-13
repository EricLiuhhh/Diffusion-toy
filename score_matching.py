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

def geometric_sequence(start, end, num):
    r = (end / start) ** (1 / (num - 1))
    sequence = [start * r**i for i in range(num)]
    return sequence


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
            nn.Linear(1, ch),
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

def train(model: nn.Module, optimizer: torch.optim.Optimizer, device='cpu', bs=2048, L=10, sigma_1=1, sigma_L=0.01, max_iters=10_000, save_every=20):
    sigmas = torch.tensor(geometric_sequence(sigma_1, sigma_L, L))
    model.to(device)
    model.train()
    for i in range(max_iters):
        x1 = torch.from_numpy(sample_x1(bs)).to(device)
        choices = torch.randint(0, L, size=(bs,))
        _sigmas = sigmas[choices].to(device)
        noise = sample_x0(bs).to(device)
        x_tilde = x1 + _sigmas[:, None] * noise
        score = model(x_tilde, _sigmas[:, None])
        loss = (torch.linalg.norm(score + (x_tilde - x1) / (_sigmas ** 2)[:, None], dim=1) * (_sigmas ** 2)).sum() / bs
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'iter: {i}, loss: {loss.item()}')

def sample(model: nn.Module, device='cpu', bs=64, T=100, L=10, sigma_1=1, sigma_L=0.01, eps=2e-5):
    sigmas = torch.tensor(geometric_sequence(sigma_1, sigma_L, L))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = sample_x0(bs).to(device)
        # move to next distribution
        for i in range(0, L):
            alpha = eps * (sigmas[i] ** 2) / (sigma_L ** 2)
            # Langevin dynamics
            for t in range(0, T):
                z = sample_x0(bs).to(device)
                _sigmas = torch.full((x.shape[0], 1), sigmas[i]).to(device)
                score = model(x, _sigmas)
                x = x + alpha / 2 * score + (alpha ** 0.5) * z
    return x.cpu().numpy()

if __name__ == "__main__":
    device = 'mps'
    model = Net()
    opti = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train(model, opti, device, max_iters=2000, sigma_1=0.1, sigma_L=0.001)
    samples = sample(model, bs=1000, sigma_1=0.1, sigma_L=0.001)

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