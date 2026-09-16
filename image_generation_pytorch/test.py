"""
Supervised capability probe for the PolyGAN generator.

Question this answers: can the generator REPRESENT the spiral curve at all,
when handed the target points directly -- no discriminator, no adversarial
game? This separates "can the network draw this shape" (representation /
conditioning) from "can the GAN train" (adversarial dynamics).

Run:
    python test.py

Output:
    - prints an MSE trace to the terminal
    - saves supervised_fit_std_True.png  (or _False.png)

Must be run from the same directory as model.py and data_loader.py.
"""

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")            # headless-safe (server / Colab)
import matplotlib.pyplot as plt

from model import Generator
from data_loader import SpiralXDataset


# ----------------------------------------------------------------------
# Config  -- the only knobs you need
# ----------------------------------------------------------------------
N_POINTS    = 2000     # bump to 10000 for a denser fit (slower on CPU)
NUM_TURNS   = 1        # 1 -> angle in [0, 2*pi], matches your training
STEPS       = 3000
LR          = 1e-3
STANDARDIZE = False     # <-- run once True, once False, compare the two PNGs
SEED        = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------
# Data: the real spiral points, ordered ALONG the curve by angle
# ----------------------------------------------------------------------
ds = SpiralXDataset(n_points=N_POINTS, num_turns=NUM_TURNS, noise=0.0)
d  = ds.data                              # [N, 2], columns = [angle, x]
d  = d[d[:, 0].argsort()]                 # sort so points run along the curve

# Standardize targets so BOTH columns matter equally to the MSE.
# Without this, the angle column's larger magnitude dominates the loss -- the
# same imbalance discussed for the GAN, but here you can SEE it: set
# STANDARDIZE = False and watch the fit neglect the small (x) column.
if STANDARDIZE:
    mean = d.mean(dim=0, keepdim=True)
    std  = d.std(dim=0, keepdim=True)
    d_fit = (d - mean) / std
else:
    mean = torch.zeros(1, 2)
    std  = torch.ones(1, 2)
    d_fit = d

d_fit = d_fit.to(device)

# Monotonic latent sweep, matching your uniform[-1, 1] z at train time.
# With z in [-1, 1], every power z^n stays bounded -- no z^12 explosion.
z = torch.linspace(-1, 1, len(d_fit)).unsqueeze(1).to(device)


# ----------------------------------------------------------------------
# Generator + plain supervised MSE fit (NO discriminator)
# ----------------------------------------------------------------------
G = Generator().to(device)
opt = torch.optim.Adam(G.parameters(), lr=LR)

print(f"device={device}  standardize={STANDARDIZE}  "
      f"points={N_POINTS}  steps={STEPS}\n")

for step in range(STEPS):
    out = G(z)
    loss = F.mse_loss(out, d_fit)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0 or step == STEPS - 1:
        print(f"step {step:4d}   mse {loss.item():.6f}")


# ----------------------------------------------------------------------
# Plot fitted curve vs targets (denormalized back to real coordinates)
# ----------------------------------------------------------------------
with torch.no_grad():
    out = G(z).cpu()

out_real = out * std + mean               # undo standardization for display
tgt_real = d.cpu()

plt.figure(figsize=(6, 6))
plt.scatter(tgt_real[:, 0], tgt_real[:, 1], s=8, alpha=0.4, label="target")
plt.scatter(out_real[:, 0], out_real[:, 1], s=8, alpha=0.7, label="fitted G(z)")
plt.legend()
plt.title(f"Supervised fit  (standardize={STANDARDIZE})")
plt.tight_layout()

fname = f"supervised_fit_std_{STANDARDIZE}.png"
plt.savefig(fname, dpi=120)
print(f"\nsaved {fname}")