"""Experimental persistence-plus-correction HBF; production HBF is unchanged."""
import numpy as np
from src.scientific_validation.v2.preflight import require
from src.scientific_validation.v2.forecast_diagnostic import LEVELS
from src.scientific_validation.v2.residual_audit import (
    flat, decompose, compare_to_baseline, fit_train_mean,
)

HEADS = ('device_prediction_head', 'state_prediction_head',
         'context_prediction_head', 'global_prediction_head')


def initialize(hbf):
    import torch
    # These four heads are Linear in the pinned implementation. Fail closed
    # if that architecture changes rather than zeroing an arbitrary module.
    for name in HEADS:
        head = getattr(hbf, name)
        require(isinstance(head, torch.nn.Linear) and head.bias is not None,
                f'Expected final linear projection: {name}')
    with torch.no_grad():
        for name in HEADS:
            head = getattr(hbf, name)
            head.weight.zero_()
            head.bias.zero_()


def forecast(model, current):
    delta = model.forecast(current)
    prediction = {k: current[k] + delta[k] for k in LEVELS}
    return prediction, delta


def train_step(model, current, target, optimizer):
    import torch
    require(all(not v.requires_grad for v in (*current.values(), *target.values())),
            'Live encoder input to pilot')
    require(all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith('hbf.')),
            'Non-HBF trainable parameters')
    expected = {id(p) for p in model.hbf.parameters()}
    actual = [p for group in optimizer.param_groups for p in group['params']]
    require(len(actual) == len(expected) and {id(p) for p in actual} == expected,
            'Optimizer must contain exactly HBF parameters')
    model.hbf.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = forecast(model, current)
    losses = model.mo(target, prediction)
    require(torch.isfinite(losses.L_HDEG).item(), 'Nonfinite training loss')
    losses.L_HDEG.backward()
    gradients = {}
    for name, p in model.hbf.named_parameters():
        require(p.grad is not None and torch.isfinite(p.grad).all().item(), f'Invalid gradient: {name}')
        gradients[name] = float(torch.linalg.vector_norm(p.grad.detach().double()).cpu())
    optimizer.step()
    require(all(torch.isfinite(p).all().item() for p in model.hbf.parameters()), 'Nonfinite updated parameter')
    return dict(losses={key: float(getattr(losses, key).detach().cpu())
                       for key in ('L_Z', 'L_S', 'L_S_tilde', 'L_G', 'L_HDEG')},
                gradient_norms=gradients)


def paired_blocks(errors, baseline):
    a, b = np.asarray(errors), np.asarray(baseline)
    require(a.shape == b.shape == (2048,), 'Expected 32 blocks of 64 errors')
    d = (a-b).reshape(32, 64).mean(axis=1)
    return dict(block_differences=d.tolist(), mean_difference=float(d.mean()),
                median_difference=float(np.median(d)), improved=int((d < 0).sum()),
                tied=int((d == 0).sum()), worsened=int((d > 0).sum()))


def correction_statistics(delta, current, target, prediction):
    d, c, t, p = map(flat, (delta, current, target, prediction))
    increment = t-c
    effective = p-c
    summary, _, _, _ = decompose(d, increment)
    energy = float(np.mean(d*d))
    increment_energy = float(np.mean(increment*increment))
    cross = float(np.mean(d*increment))
    denom = np.sqrt(energy*increment_energy)
    return dict(correction_energy=energy, true_increment_energy=increment_energy,
                correction_increment_inner_product=cross,
                uncentered_alignment=float(np.clip(cross/denom, -1, 1)) if denom else None,
                uncentered_alignment_status='defined' if denom else 'undefined_zero_energy',
                centered_alignment=summary['centered_alignment'],
                centered_alignment_status=summary['centered_alignment_status'],
                correction_variance=summary['prediction_variance'],
                increment_variance=summary['target_variance'],
                effective_correction_energy=float(np.mean(effective*effective)),
                addition_rounding_mse=float(np.mean((effective-d)**2)))


def evaluate(model, cache, device, means, *, step_zero=False):
    import torch
    model.eval()
    errors, summary = {}, {}
    for split in ('train', 'val'):
        predictions, corrections = {k: [] for k in LEVELS}, {k: [] for k in LEVELS}
        with torch.no_grad():
            for start in range(0, 2048, 32):
                current = {k: v[start:start+32].to(device) for k, v in cache[split]['current'].items()}
                prediction, delta = forecast(model, current)
                for k in LEVELS:
                    require(prediction[k].shape == current[k].shape and
                            torch.isfinite(prediction[k]).all().item() and torch.isfinite(delta[k]).all().item(),
                            f'Invalid residual prediction: {k}')
                    if step_zero:
                        require(torch.count_nonzero(delta[k]).item() == 0 and torch.equal(prediction[k], current[k]),
                                f'Initialization does not exactly reproduce persistence: {k}')
                    predictions[k].append(prediction[k].cpu().numpy().copy())
                    corrections[k].append(delta[k].cpu().numpy().copy())
        levels = {}
        collected = {pred: [] for pred in ('residual_hbf', 'persistence', 'train_mean')}
        centered_errors = []
        for k in LEVELS:
            c = cache[split]['current'][k].numpy()
            t = cache[split]['target'][k].numpy()
            p, d = np.concatenate(predictions[k]), np.concatenate(corrections[k])
            mean_prediction = np.broadcast_to(means[k], flat(t).shape)
            details, per = {}, {}
            for pred, values in (('residual_hbf', p), ('persistence', c), ('train_mean', mean_prediction)):
                s, a, centered, _ = decompose(flat(values), flat(t))
                details[pred], per[pred] = s, a
                collected[pred].append(a)
                if pred == 'residual_hbf':
                    centered_errors.append(centered)
            details['comparisons'] = {pred: compare_to_baseline(per['residual_hbf'], per[pred])
                                      for pred in ('persistence', 'train_mean')}
            details['paired_blocks'] = {pred: paired_blocks(per['residual_hbf'], per[pred])
                                       for pred in ('persistence', 'train_mean')}
            details['correction'] = correction_statistics(d, c, t, p)
            levels[k] = details
        for pred, arrays in collected.items():
            errors[f'{split}_{pred}_mse'] = np.stack(arrays, axis=1)
        errors[f'{split}_residual_hbf_centered_mse'] = np.stack(centered_errors, axis=1)
        summary[split] = dict(paired_windows=2048, levels=levels)
    return summary, errors
