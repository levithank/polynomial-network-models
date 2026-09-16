import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Generator

# -----------------------------
# 1. Load pretrained generator
# -----------------------------
ckpt_path ="/mnt/c/Users/leroy/Documents/GitHub/polynomial_netsv2/image_generation_pytorch/experiments/circle_run_2_no_activation_1108_122050/models/generator-100.pkl"  
# change this if needed must be in linux format

generator = Generator()
generator.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
generator.eval()

# -----------------------------
# 2. Create z inputs in [-10, 10]
# -----------------------------
z = torch.linspace(-1, 1, 500).unsqueeze(1)   # shape [500, 1]

# -----------------------------
# 3. Run generator
# -----------------------------
with torch.no_grad():
    outputs = generator(z).cpu().numpy()

z_np = z.squeeze(1).cpu().numpy()
x_out = outputs[:, 0]
y_out = outputs[:, 1]

x=np.cos(np.pi*z_np)
y=np.sin(np.pi*z_np)

# -----------------------------
# 4. Plot z -> x
# -----------------------------
plt.figure(figsize=(6,4))
#appprox curve
plt.plot(z_np, x_out)
#default curve
plt.plot(z_np,x)

plt.plot()
plt.xlabel("z")
plt.ylabel("x(z)")
plt.title("Generator Output: x vs z")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Plot z -> y
# -----------------------------
plt.figure(figsize=(6,4))
#approx curve
plt.plot(z_np, y_out)
#default sin curve
plt.plot(z_np,-y)
plt.xlabel("z")
plt.ylabel("y(z)")
plt.title("Generator Output: y vs z")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Plot x-> y
# -----------------------------
plt.figure(figsize=(6,4))
#approx curve
plt.plot(x_out, y_out)
#default sin curve
plt.plot(x,-y)
plt.xlabel("z")
plt.ylabel("y(z)")
plt.title("Generator Output: y vs z")
plt.grid(True)
plt.tight_layout()
plt.show()