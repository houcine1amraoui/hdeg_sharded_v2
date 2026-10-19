"""Bounded autograd diagnostics. No optimizer, backward accumulation or updates."""
import numpy as np
from src.scientific_validation.v2.preflight import require

BLOCK_IDS = (0, 10, 21, 31)
BATCH_SIZE = 32
LOSS_KEYS = ('L_Z', 'L_S', 'L_S_tilde', 'L_G', 'L_HDEG')
BRANCHES = ('full', 'prediction', 'target')
MODULES = ('dbrl', 'bse', 'bil', 'ebrl', 'hbf')
IDENTITY_RTOL = 1e-4


def summarize_gradient(value):
    """None is absent; a connected all-zero tensor is a distinct finding."""
    if value is None:
        return dict(status='absent', elements=0, nonfinite_elements=0,
                    l2=None, rms=None, max_abs=None)
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    bad = int((~np.isfinite(a)).sum())
    if bad:
        return dict(status='nonfinite', elements=len(a), nonfinite_elements=bad,
                    l2=None, rms=None, max_abs=None)
    norm = float(np.linalg.norm(a))
    return dict(status='zero' if not np.any(a != 0) else 'nonzero', elements=len(a),
                nonfinite_elements=0, l2=norm, rms=norm/np.sqrt(len(a)),
                max_abs=float(np.max(np.abs(a))))


def compare_vectors(full, prediction, target):
    """All inputs are flattened float64 vectors; missing paths already zero-filled."""
    f, p, t = [np.asarray(x, dtype=np.float64) for x in (full, prediction, target)]
    require(f.shape == p.shape == t.shape, 'Gradient comparison shape mismatch')
    if not all(np.isfinite(x).all() for x in (f, p, t)):
        return dict(status='NOT_EVALUATED_NONFINITE')
    fn, pn, tn = [float(np.linalg.norm(x)) for x in (f, p, t)]
    summed = p+t
    sn = float(np.linalg.norm(summed))
    residual = float(np.linalg.norm(f-summed))
    scale = fn+pn+tn
    relative = residual/scale if scale else 0.0
    cosine = float(np.clip(np.dot(p, t)/(pn*tn), -1, 1)) if pn and tn else None
    return dict(status='PASS' if relative <= IDENTITY_RTOL else 'FAIL',
                full_l2=fn, prediction_l2=pn, target_l2=tn,
                sum_l2=sn, dot_product=float(np.dot(p, t)), cosine=cosine,
                cosine_status='defined' if cosine is not None else 'undefined_zero_branch_norm',
                sum_to_branch_norm_ratio=sn/(pn+tn) if pn+tn else None,
                cancellation_fraction=float(np.clip(1-sn/(pn+tn), 0, 1)) if pn+tn else None,
                decomposition_residual_l2=residual, decomposition_relative_residual=relative,
                decomposition_tolerance=IDENTITY_RTOL)


def group_indices(names):
    groups = {m: [i for i, n in enumerate(names) if n.split('.')[0] == m] for m in MODULES}
    require(all(groups.values()), 'Expected trainable parameters in all five modules')
    require(sum(map(len, groups.values())) == len(names), 'Unclassified trainable parameter')
    groups['encoder'] = [i for i, n in enumerate(names) if not n.startswith('hbf.')]
    groups['all'] = list(range(len(names)))
    return groups


def group_summary(values, indices):
    summaries = [summarize_gradient(values[i]) for i in indices]
    counts = {status: sum(s['status'] == status for s in summaries)
              for status in ('absent', 'zero', 'nonzero', 'nonfinite')}
    finite = counts['nonfinite'] == 0
    norm = float(np.sqrt(sum(s['l2']**2 for s in summaries if s['l2'] is not None))) if finite else None
    return dict(parameter_tensors=len(indices), counts=counts, l2=norm,
                nonfinite_elements=sum(s['nonfinite_elements'] for s in summaries))


def vector(values, indices, sizes):
    return np.concatenate([np.zeros(sizes[i], dtype=np.float64) if values[i] is None
                           else values[i].reshape(-1) for i in indices])


def audit_batch(model, x, y):
    """One shared forward graph; 15 reverse-mode queries (3 branches x 5 losses).

    Detach is ONLY used to isolate derivative paths for measurement. The full
    objective and model are unchanged. No optimizer is constructed.
    """
    import torch
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    names, params = zip(*named)
    sizes = [p.numel() for p in params]
    groups = group_indices(names)
    require(all(p.grad is None for p in params), 'Unexpected accumulated parameter gradients')
    current = model.encode_window(x)
    target = model.encode_window(y)
    prediction = model.forecast(current)
    objectives = {
        'full': model.mo(target, prediction),
        'prediction': model.mo({k: v.detach() for k, v in target.items()}, prediction),
        'target': model.mo(target, {k: v.detach() for k, v in prediction.items()}),
    }
    results = {}
    full_levels = []
    for loss_index, loss_name in enumerate(LOSS_KEYS):
        gradients, branch_losses = {}, {}
        for branch_index, branch in enumerate(BRANCHES):
            loss = getattr(objectives[branch], loss_name)
            require(loss.ndim == 0 and torch.isfinite(loss).item(), 'Nonfinite or non-scalar loss')
            branch_losses[branch] = float(loss.detach().cpu())
            require(loss.requires_grad, f'Loss is disconnected from autograd: {loss_name}/{branch}')
            last = loss_index == len(LOSS_KEYS)-1 and branch_index == len(BRANCHES)-1
            raw = torch.autograd.grad(loss, params, allow_unused=True,
                                      retain_graph=not last, create_graph=False)
            gradients[branch] = [None if g is None else g.detach().to('cpu', dtype=torch.float64).numpy()
                                 for g in raw]
            del raw
        require(branch_losses['full'] == branch_losses['prediction'] == branch_losses['target'],
                'Branch-isolation changed forward loss values')
        details = {name: {branch: summarize_gradient(gradients[branch][i]) for branch in BRANCHES}
                   for i, name in enumerate(names)}
        summary = {}
        for group, indices in groups.items():
            summary[group] = dict(
                branches={branch: group_summary(gradients[branch], indices) for branch in BRANCHES},
                comparison=compare_vectors(*(vector(gradients[branch], indices, sizes) for branch in BRANCHES)))
        results[loss_name] = dict(loss=branch_losses['full'], groups=summary, parameters=details)
        full_vector = vector(gradients['full'], groups['all'], sizes)
        if loss_name != 'L_HDEG':
            full_levels.append(full_vector)
        else:
            weights = [float(getattr(model.mo, key).detach().cpu())
                       for key in ('lambda_Z', 'lambda_S', 'lambda_S_tilde', 'lambda_G')]
            weighted = sum(w*g for w, g in zip(weights, full_levels))
            weight_identity = compare_vectors(full_vector, weighted, np.zeros_like(weighted))
        del gradients
    require(all(p.grad is None for p in params), 'autograd.grad unexpectedly accumulated parameter gradients')
    # Module absences in level/branch-specific objectives are reported, not
    # automatically classified as defects; e.g. target-only loss cannot reach HBF.
    return dict(losses=results, loss_weights=weights,
                weighted_loss_gradient_identity=weight_identity,
                parameter_grad_buffers_remain_none=True)


def interpretation_flags(batch_results):
    nonfinite, failed_identities, absent_full, zero_full = [], [], [], []
    for batch in batch_results:
        bid = batch['block_id']
        for loss_name, loss in batch['gradients']['losses'].items():
            for name, branches in loss['parameters'].items():
                for branch, value in branches.items():
                    if value['status'] == 'nonfinite':
                        nonfinite.append(f'block={bid}/{loss_name}/{branch}/{name}')
                if loss_name == 'L_HDEG':
                    if branches['full']['status'] == 'absent':
                        absent_full.append(f'block={bid}/{name}')
                    elif branches['full']['status'] == 'zero':
                        zero_full.append(f'block={bid}/{name}')
            if loss['groups']['all']['comparison']['status'] == 'FAIL':
                failed_identities.append(f'block={bid}/{loss_name}/branch_sum')
        if batch['gradients']['weighted_loss_gradient_identity']['status'] == 'FAIL':
            failed_identities.append(f'block={bid}/weighted_loss_sum')
    return dict(nonfinite_gradients=nonfinite, failed_gradient_identities=failed_identities,
                absent_full_total_gradients=absent_full, zero_full_total_gradients=zero_full,
                interpretation='Local derivative evidence only; no convergence, update-history or causal-collapse conclusion')
