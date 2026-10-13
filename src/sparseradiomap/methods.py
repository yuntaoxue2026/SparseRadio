"""Algorithms accept ONLY shape, observed indices/values, config and seed."""
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.linalg import cg
import torch
from torch import nn
from threadpoolctl import threadpool_limits

def device_name():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def idw(shape, rc, values, k=8, power=2.0):
    target = np.indices(shape).reshape(2, -1).T
    distances, inds = cKDTree(rc).query(target, k=min(k, len(rc)), workers=1)
    if distances.ndim == 1:
        distances, inds = distances[:, None], inds[:, None]
    weights = np.maximum(distances, 1e-12)**(-power)
    pred = np.sum(weights*values[inds], axis=1)/weights.sum(axis=1)
    pred[np.ravel_multi_index(rc.T, shape)] = values
    return pred.reshape(shape)

def grid_laplacian(shape):
    h, w = shape
    def lap1(n):
        d = np.full(n, 2.0)
        d[[0, -1]] = 1
        return sparse.diags([-np.ones(n-1), d, -np.ones(n-1)], [-1, 0, 1], format='csr')
    return sparse.kronsum(lap1(w), lap1(h), format='csr')

def smooth(shape, rc, values, lam=0.1, rtol=1e-5, maxiter=2500):
    # Sum observed squared residuals + lambda * sum 4-neighbor edge differences².
    # Natural (no outgoing edge / Neumann) boundary; plain grid across buildings.
    n = np.prod(shape)
    inds = np.ravel_multi_index(rc.T, shape)
    diagonal = np.zeros(n)
    diagonal[inds] = 1
    a = sparse.diags(diagonal, format='csr') + lam*grid_laplacian(shape)
    b = np.zeros(n)
    b[inds] = values
    preconditioner = sparse.diags(1/a.diagonal())
    with threadpool_limits(limits=1):
        sol, info = cg(a, b, x0=np.full(n, values.mean()), M=preconditioner,
                       rtol=rtol, atol=0, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f'CG did not converge: info={info}')
    return sol.reshape(shape), {'cg_relative_residual': float(np.linalg.norm(a@sol-b)/np.linalg.norm(b))}

def lowrank(shape, rc, values, seed, rank=8, steps=200, lr=0.03,
            reg=0.001, device='auto'):
    dev = device_name() if device == 'auto' else device
    torch.manual_seed(seed)
    h, w = shape
    if rank < 2:
        raise ValueError('rank includes one fixed constant component, must be >=2')
    # One fixed rank-one constant component plus rank-1 learned residual factors.
    # Unobserved rows/columns shrink to the observed mean rather than to zero.
    mean = float(values.mean())
    u = nn.Parameter(0.03*torch.randn(h, rank-1, device=dev))
    v = nn.Parameter(0.03*torch.randn(w, rank-1, device=dev))
    indices = torch.as_tensor(rc, dtype=torch.long, device=dev)
    y = torch.as_tensor(values, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam([u, v], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        fitted = mean + (u[indices[:, 0]]*v[indices[:, 1]]).sum(1)
        loss = (fitted-y).square().mean() + reg*(u.square().mean()+v.square().mean())
        loss.backward()
        opt.step()
    with torch.no_grad():
        raw = (mean + u@v.T).cpu().numpy()
    return raw, {'device': dev, 'final_observation_objective': float(loss.detach().cpu())}

def positional(coords, bands):
    frequencies = 2.0**torch.arange(bands, device=coords.device)*torch.pi
    z = (coords[..., None]*frequencies).reshape(len(coords), -1)
    return torch.cat([coords, torch.sin(z), torch.cos(z)], dim=1)

def mlp(shape, rc, values, seed, bands=4, width=64, steps=200,
        lr=0.003, reg=1e-5, batch_size=1024, device='auto'):
    dev = device_name() if device == 'auto' else device
    torch.manual_seed(seed)
    scale = torch.tensor(np.array(shape)-1, dtype=torch.float32, device=dev)
    coords = torch.as_tensor(rc, dtype=torch.float32, device=dev)/scale*2-1
    x = positional(coords, bands)
    y = torch.as_tensor(values, dtype=torch.float32, device=dev)
    model = nn.Sequential(nn.Linear(2+4*bands, width), nn.ReLU(),
                          nn.Linear(width, width), nn.ReLU(),
                          nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    # Index sampling stays on CPU for stable seeded RNG across backends.
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        batch = torch.as_tensor(rng.choice(len(y), min(batch_size, len(y)), replace=False), device=dev)
        opt.zero_grad(set_to_none=True)
        loss = (model(x[batch]).squeeze(1)-y[batch]).square().mean()
        loss.backward()
        opt.step()
    target = np.indices(shape).reshape(2, -1).T
    parts = []
    with torch.no_grad():
        for c in np.array_split(target, max(1, len(target)//4096)):
            z = torch.as_tensor(c, dtype=torch.float32, device=dev)/scale*2-1
            parts.append(model(positional(z, bands)).squeeze(1).cpu().numpy())
    return np.concatenate(parts).reshape(shape), {'device': dev, 'final_minibatch_mse': float(loss.detach().cpu())}

def reconstruct(method, shape, rc, values, cfg, seed):
    rc = np.asarray(rc, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    if len(rc) != len(values) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid observations')
    torch.set_num_threads(1)
    if method == 'idw':
        return idw(shape, rc, values, **cfg), {'device': 'cpu'}
    if method == 'smooth':
        pred, meta = smooth(shape, rc, values, **cfg)
        return pred, dict(meta, device='cpu')
    if method == 'lowrank':
        return lowrank(shape, rc, values, seed, **cfg)
    if method == 'mlp':
        return mlp(shape, rc, values, seed, **cfg)
    raise ValueError(method)
