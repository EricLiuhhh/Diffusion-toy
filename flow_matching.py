import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdyn.core import NeuralODE
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

def train(model: nn.Module, optimizer: torch.optim.Optimizer, device='cpu', bs=2048, L=10, sigma=0.01, max_iters=10_000, save_every=20):
    model.to(device)
    model.train()
    for i in range(max_iters):
        x0 = sample_x0(bs).to(device)
        x1 = torch.from_numpy(sample_x1(bs)).to(device)
        t = torch.rand((bs,), device=device)[:, None].to(device)
        xt = (t * x1 + (1 - t) * x0) + sigma * sample_x0(bs).to(device)
        vt = model(xt, t)
        ut = x1 - x0
        loss = ((vt - ut) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'iter: {i}, loss: {loss.item()}')

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(x, t.repeat(x.shape[0])[:, None])
    
def sample(model: nn.Module, device='cpu', bs=64):
    model = model.to(device)
    model.eval()
    with torch.no_grad():

        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        with torch.no_grad():
            traj = node.trajectory(
                sample_x0(bs).to(device),
                t_span=torch.linspace(0, 1, 100).to(device),
            )

    return traj[-1].cpu().numpy()

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