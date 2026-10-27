import os, math, torch
import torch.nn as nn

# 1. Device Allocation
d1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
d2 = torch.device("cuda:2" if torch.cuda.device_count() > 2 else ("cuda:0" if torch.cuda.is_available() else "cpu"))

# 2. Simulation: 1D Poisson -u''(x) = f(x), u(0)=u(1)=0 (2000 Samples)
N, M, P = 2000, 32, 101
dx = 1.0 / (P - 1)
torch.manual_seed(2026)

xg = torch.linspace(0.0, 1.0, P)
xs = torch.linspace(0.0, 1.0, M)
k = torch.tensor([1.0, 2.0, 3.0])
c = torch.empty((N, 3)).uniform_(-2.0, 2.0)

fs = torch.matmul(c, torch.sin(math.pi * torch.outer(xs, k)).T)
fg = torch.matmul(c, torch.sin(math.pi * torch.outer(xg, k)).T)
ug = torch.matmul(c / (k * math.pi)**2, torch.sin(math.pi * torch.outer(xg, k)).T)
uxg = torch.matmul(c / (k * math.pi), torch.cos(math.pi * torch.outer(xg, k)).T)

# Save Dataset
torch.save({
    "f_sensors": fs, "f_grid": fg, "u_grid": ug, "ux_grid": uxg, "x_grid": xg
}, os.path.expanduser("~/poisson_dataset_2000.pt"))

# 80:20 Split
tr = 1600
fs_tr, fs_te = fs[:tr], fs[tr:]
ug_tr, ug_te = ug[:tr], ug[tr:]
uxg_tr, uxg_te = uxg[:tr], uxg[tr:]
fg_tr = fg[:tr]

# 3. Model Architecture
class DeepONet(nn.Module):
    def __init__(self, m=32, p=64):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(m, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, p))
        self.trunk = nn.Sequential(nn.Linear(1, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, p))
        self.b0 = nn.Parameter(torch.zeros(1))
    def forward(self, f, x):
        out = torch.matmul(self.branch(f), self.trunk(x.unsqueeze(-1)).T) + self.b0
        return x * (1.0 - x) * out

net_d = DeepONet(M).to(d1)
net_p = DeepONet(M).to(d2)
opt_d = torch.optim.Adam(net_d.parameters(), lr=1e-3)
opt_p = torch.optim.Adam(net_p.parameters(), lr=1e-3)

# 4. Training Loop (300 Epochs)
epochs = 300
lh_d, lh_p = [], []
f_d, u_d, x_d = fs_tr.to(d1), ug_tr.to(d1), xg.to(d1)
f_p, fg_p, x_p = fs_tr.to(d2), fg_tr.to(d2), xg.to(d2)

for _ in range(epochs):
    # DeepONet (Supervised)
    p_d = net_d(f_d, x_d)
    loss_d = torch.mean((p_d - u_d)**2)
    opt_d.zero_grad(); loss_d.backward(); opt_d.step()
    lh_d.append(loss_d.item())

    # PINN (PDE Residual)
    p_p = net_p(f_p, x_p)
    d2u = (p_p[:, 2:] - 2.0 * p_p[:, 1:-1] + p_p[:, :-2]) / (dx**2)
    loss_p = torch.mean((-d2u - fg_p[:, 1:-1])**2)
    opt_p.zero_grad(); loss_p.backward(); opt_p.step()
    lh_p.append(loss_p.item())

# 5. Relative Sobolev H1 Evaluation
def rel_h1(u_pred, u_true, ux_true):
    ux_pred = torch.gradient(u_pred, spacing=dx, dim=1)[0]
    num = torch.sqrt(torch.mean((u_true - u_pred)**2) + torch.mean((ux_true - ux_pred)**2))
    den = torch.sqrt(torch.mean(u_true**2) + torch.mean(ux_true**2))
    return (num / den).item()

net_d.eval(); net_p.eval()
with torch.no_grad():
    h1_d_tr = rel_h1(net_d(f_d, x_d).cpu(), ug_tr, uxg_tr)
    h1_d_te = rel_h1(net_d(fs_te.to(d1), x_d).cpu(), ug_te, uxg_te)
    h1_p_tr = rel_h1(net_p(f_p, x_p).cpu(), ug_tr, uxg_tr)
    h1_p_te = rel_h1(net_p(fs_te.to(d2), x_p).cpu(), ug_te, uxg_te)

# 6. Save Metrics
with open(os.path.expanduser("~/metrics.txt"), "w") as f:
    f.write("Model\tTrain_Rel_H1\tTest_Rel_H1\n")
    f.write(f"DeepONet\t{h1_d_tr:.6e}\t{h1_d_te:.6e}\n")
    f.write(f"PINN\t{h1_p_tr:.6e}\t{h1_p_te:.6e}\n")

# 7. Save SVG Training Curves
w, h, pad = 800, 400, 50
all_min = min(min(lh_d), min(lh_p))
all_max = max(max(lh_d), max(lh_p))
def to_pts(data):
    pts = []
    for i, v in enumerate(data):
        x = pad + (i / (epochs - 1)) * (w - 2 * pad)
        norm = (math.log10(v) - math.log10(all_min)) / (math.log10(all_max) - math.log10(all_min) + 1e-6)
        y = (h - pad) - norm * (h - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)

svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#0f172a; font-family:monospace;">',
    f'<rect x="{pad}" y="{pad}" width="{w-2*pad}" height="{h-2*pad}" fill="#1e293b" stroke="#334155"/>',
    f'<text x="{w/2}" y="30" fill="#f8fafc" font-size="14" font-weight="bold" text-anchor="middle">Log Loss: DeepONet vs PINN</text>',
    f'<polyline points="{to_pts(lh_d)}" fill="none" stroke="#38bdf8" stroke-width="2"/>',
    f'<polyline points="{to_pts(lh_p)}" fill="none" stroke="#f43f5e" stroke-width="2"/>',
    f'<text x="{w-180}" y="{pad+25}" fill="#38bdf8" font-size="12">-- DeepONet</text>',
    f'<text x="{w-180}" y="{pad+45}" fill="#f43f5e" font-size="12">-- PINN</text>',
    '</svg>'
]
with open(os.path.expanduser("~/training_curves.svg"), "w") as f:
    f.write("\n".join(svg))

print("[+] Execution Complete.")