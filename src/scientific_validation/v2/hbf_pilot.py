"""Fixed-budget HBF-only pilot helpers; encoder representations are constants."""
import hashlib
import numpy as np
from src.scientific_validation.v2.preflight import require
from src.scientific_validation.v2.forecast_diagnostic import LEVELS, window_mse, error_summary, Moments

STEPS = 640
EVALUATE_AT = (0, 64, 128, 320, 640)
LR = 1e-4


def update_order():
    rng = np.random.Generator(np.random.PCG64(42))
    return np.concatenate([rng.permutation(2048).reshape(64, 32) for _ in range(10)])


def cache_digest(cache):
    h = hashlib.sha256()
    for split in ('train', 'val'):
        for role in ('current', 'target'):
            for level in LEVELS:
                a = cache[split][role][level].numpy()
                h.update(f'{split}/{role}/{level}/{a.shape}/{a.dtype}'.encode())
                h.update(a.tobytes())
    return h.hexdigest()


def train_step(model, current, target, optimizer):
    """Inputs must be detached frozen representations; only HBF is trainable."""
    import torch
    require(all(not v.requires_grad for v in (*current.values(), *target.values())), 'Live encoder input to pilot')
    require(all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith('hbf.')),
            'Non-HBF trainable parameters')
    expected_ids = {id(p) for p in model.hbf.parameters()}
    actual = [p for group in optimizer.param_groups for p in group['params']]
    require(len(actual) == len(expected_ids) and {id(p) for p in actual} == expected_ids,
            'Optimizer must contain exactly HBF parameters')
    model.hbf.train()
    optimizer.zero_grad(set_to_none=True)
    predicted = model.forecast(current)
    losses = model.mo(target, predicted)
    require(torch.isfinite(losses.L_HDEG).item(), 'Nonfinite training loss')
    losses.L_HDEG.backward()
    for name, value in model.hbf.named_parameters():
        require(value.grad is not None and torch.isfinite(value.grad).all().item(), f'Invalid HBF gradient: {name}')
    optimizer.step()
    require(all(torch.isfinite(p).all().item() for p in model.hbf.parameters()), 'Nonfinite updated HBF parameter')
    return {key: float(getattr(losses, key).detach().cpu())
            for key in ('L_Z', 'L_S', 'L_S_tilde', 'L_G', 'L_HDEG')}


def evaluate(model, cache, device):
    import torch
    model.eval()
    errors, summary = {}, {}
    for split in ('train', 'val'):
        h, p = [], []
        moments = {level: {role: Moments() for role in ('current', 'target', 'hbf')} for level in LEVELS}
        block_variances = {level: {role: [] for role in ('current', 'target', 'hbf')} for level in LEVELS}
        block_outputs = {level: [] for level in LEVELS}
        with torch.no_grad():
            for start in range(0, 2048, 32):
                current = {k: v[start:start+32].to(device) for k,v in cache[split]['current'].items()}
                predicted = model.forecast(current)
                hh, pp = [], []
                for level in LEVELS:
                    c = cache[split]['current'][level][start:start+32].numpy()
                    t = cache[split]['target'][level][start:start+32].numpy()
                    f = predicted[level].detach().cpu().numpy()
                    hh.append(window_mse(f, t)); pp.append(window_mse(c, t))
                    for role, a in (('current', c), ('target', t), ('hbf', f)):
                        moments[level][role].update(a)
                    block_outputs[level].append(f.copy())
                    if (start+32) % 64 == 0:
                        for role, a in (('current', cache[split]['current'][level][start-32:start+32].numpy()),
                                        ('target', cache[split]['target'][level][start-32:start+32].numpy()),
                                        ('hbf', np.concatenate(block_outputs[level]))):
                            m = Moments(); m.update(a)
                            block_variances[level][role].append(m.summary()['centered_population_variance'])
                        block_outputs[level] = []
                h.append(np.stack(hh, axis=1)); p.append(np.stack(pp, axis=1))
        h, p = np.concatenate(h), np.concatenate(p)
        errors[split+'_hbf_mse'], errors[split+'_persistence_mse'] = h, p
        levels = {}
        for j, level in enumerate(LEVELS):
            bh, bp = h[:, j].reshape(32, 64).mean(1), p[:, j].reshape(32, 64).mean(1)
            levels[level] = dict(errors=error_summary(h[:, j], p[:, j]),
                block_hbf_mse=bh.tolist(), block_persistence_mse=bp.tolist(),
                block_comparison=dict(hbf_lower=int((bh<bp).sum()), exactly_tied=int((bh==bp).sum()), hbf_higher=int((bh>bp).sum())),
                variation={role:m.summary() for role,m in moments[level].items()},
                within_block_variance=block_variances[level])
        summary[split] = dict(paired_windows=2048, levels=levels)
    return summary, errors
