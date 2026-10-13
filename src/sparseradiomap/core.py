import numpy as np

def decode_gray(gray):
    """Saved 2D PNG pathgain: normalized g/255, -147 + 100*g/255 dB.

    Zero is a saved floor, NOT the exact pathgain before truncation.
    """
    return np.asarray(gray, dtype=np.float64) / 255.0

def to_db(normalized):
    return -147.0 + 100.0 * np.asarray(normalized)

def sampling(valid, fraction, mechanism, seed, hole_size=64):
    """Sample exact floor(p*Nvalid) distinct points; closed-open in-bounds hole.

    Rectangle RNG is separate from sampling RNG, so the hole is shared across
    fractions for a given image/seed. Paired conditions have equal counts.
    """
    valid = np.asarray(valid, dtype=bool)
    h, w = valid.shape
    hole = np.zeros_like(valid)
    bounds = None
    if mechanism == 'block':
        if hole_size >= min(h, w):
            raise ValueError('Hole must fit with a nonempty exterior')
        rng_rect = np.random.default_rng(np.random.SeedSequence([seed, 991]))
        r = int(rng_rect.integers(0, h-hole_size+1))
        c = int(rng_rect.integers(0, w-hole_size+1))
        bounds = [r, c, r+hole_size, c+hole_size]
        hole[r:r+hole_size, c:c+hole_size] = True
    elif mechanism != 'random':
        raise ValueError(mechanism)
    pool = np.flatnonzero(valid & ~hole)
    n = max(1, int(np.floor(fraction*valid.sum())))
    if len(pool) < n:
        raise ValueError('Insufficient exterior: do not silently change sampling')
    rng = np.random.default_rng(seed)
    obs = np.zeros(valid.size, dtype=bool)
    obs[rng.choice(pool, size=n, replace=False)] = True
    return obs.reshape(valid.shape), hole, bounds

def high_gradient(y, valid, quantile=0.9):
    """Top decile of central/one-sided gradient, only outdoor stencils.

    Exclude any central stencil touching buildings. Ground truth is used here
    solely for scoring, never for fitting or selecting test checkpoints.
    """
    interior = valid.copy()
    interior[1:] &= valid[:-1]
    interior[:-1] &= valid[1:]
    interior[:, 1:] &= valid[:, :-1]
    interior[:, :-1] &= valid[:, 1:]
    gy, gx = np.gradient(y)
    g = np.hypot(gy, gx)
    threshold = np.quantile(g[interior], quantile)
    return interior & (g >= threshold), float(threshold)

def metrics(raw, truth, valid, obs, hole):
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite prediction')
    pred = np.clip(raw, 0, 1)
    hidden = valid & ~obs
    grad, threshold = high_gradient(truth, valid)
    e = pred-truth
    out = {'n_valid': int(valid.sum()), 'n_observed': int(obs.sum()),
           'n_hidden': int(hidden.sum()), 'n_hole': int((hole & hidden).sum()),
           'gradient_threshold': threshold,
           'raw_min': float(raw.min()), 'raw_max': float(raw.max()),
           'clip_fraction': float(((raw < 0) | (raw > 1))[valid].mean())}
    for prefix, mask in [('hidden', hidden), ('observed', obs),
                         ('hole', hidden & hole), ('gradient', hidden & grad)]:
        z = e[mask]
        for name, value in [('mae', np.mean(np.abs(z)) if len(z) else np.nan),
                            ('rmse', np.sqrt(np.mean(z*z)) if len(z) else np.nan),
                            ('bias', np.mean(z) if len(z) else np.nan)]:
            out[prefix+'_'+name] = float(value)
            out[prefix+'_'+name+'_db'] = float(value*100)
    z = (raw-truth)[hidden]
    out['raw_hidden_rmse'] = float(np.sqrt(np.mean(z*z)))
    return out
