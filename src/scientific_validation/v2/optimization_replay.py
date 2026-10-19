"""Instrumentation only: no alternate objective, architecture, or optimizer."""
import hashlib
import numpy as np
from src.scientific_validation.v2.preflight import require
from src.scientific_validation.v2.forecast_diagnostic import LEVELS
from src.scientific_validation.v2.residual_hbf_pilot import HEADS
from src.scientific_validation.v2.residual_audit import flat, centered, decompose, compare_to_baseline

STEPS = 8
PROBE_ROWS = 64


def cache_digest(cache):
    h = hashlib.sha256()
    for role in ('current', 'target'):
        for level in LEVELS:
            a = cache['train'][role][level].numpy()
            h.update(f'{role}/{level}/{a.shape}/{a.dtype}'.encode())
            h.update(a.tobytes())
    return h.hexdigest()


def variation(a):
    a = flat(a)
    mean, deviations = centered(a)
    variance = float(np.mean(deviations**2))
    energy = float(np.mean(a*a))
    return dict(rows=len(a), coordinates=a.shape[1], rms=float(np.sqrt(energy)),
                mean_square=float(np.mean(mean**2)), centered_variance=variance,
                variance_energy_ratio=variance/energy if energy else None,
                varying_coordinates=int(np.any(deviations != 0, axis=0).sum()))


def forward(model, current):
    """Capture the actual shared input to all four final heads, without changing it."""
    import torch
    captured = {}
    handles = []
    for name in HEADS:
        def capture(module, args, name=name):
            captured[name] = args[0].detach().clone()
        handles.append(getattr(model.hbf, name).register_forward_pre_hook(capture))
    try:
        delta = model.forecast(current)
    finally:
        for handle in handles:
            handle.remove()
    require(set(captured) == set(HEADS), 'Missing head activation')
    shared = captured[HEADS[0]]
    require(all(torch.equal(shared, v) for v in captured.values()), 'Heads did not share identical features')
    prediction = {k: current[k]+delta[k] for k in LEVELS}
    require(all(torch.isfinite(v).all().item() for v in (*prediction.values(), shared)), 'Nonfinite forward')
    return prediction, delta, shared


def contributions(weight_output, bias, delta):
    """Float64 linear-head decomposition; fused-float32 rounding is reported separately."""
    w, d = flat(weight_output), flat(delta)
    b = np.broadcast_to(np.asarray(bias, dtype=np.float64), w.shape)
    total = w+b
    return dict(weight=variation(w), bias_rms=float(np.sqrt(np.mean(b*b))),
                bias_energy=float(np.mean(b*b)), weight_bias_cross_term=float(2*np.mean(w*b)),
                reconstructed_energy=float(np.mean(total*total)),
                actual_correction=variation(d),
                fused_head_rounding_mse=float(np.mean((d-total)**2)))


def summarize(model, current, target, prediction, delta, shared):
    h = shared.detach().cpu().double().numpy()
    levels, error_arrays = {}, []
    inputs = np.concatenate([flat(current[k].detach().cpu().numpy()) for k in LEVELS], axis=1)
    for k, name in zip(LEVELS, HEADS):
        c, t, p, d = [v[k].detach().cpu().numpy() for v in (current,target,prediction,delta)]
        head = getattr(model.hbf, name)
        weight = head.weight.detach().cpu().double().numpy()
        bias = head.bias.detach().cpu().double().numpy()
        forecast, errors, _, _ = decompose(p, t)
        persistence, baseline, _, _ = decompose(c, t)
        levels[k] = dict(forecast=forecast, persistence=persistence,
            comparison=compare_to_baseline(errors, baseline),
            head=contributions(h @ weight.T, bias, d),
            increment=variation(flat(t)-flat(c)),
            addition_rounding_mse=float(np.mean((flat(p)-flat(c)-flat(d))**2)))
        error_arrays.append(np.stack((errors, baseline), axis=1))
    return dict(input_hierarchy=variation(inputs), shared_head_input=variation(h), levels=levels), np.stack(error_arrays, axis=1)


def parameter_updates(model, before, optimizer):
    """Measure actual displacement after step; inspect Adam's stored second moment."""
    import torch
    result = {}
    group = optimizer.param_groups[0]
    beta2, eps = group['betas'][1], group['eps']
    for name, p in model.hbf.named_parameters():
        old = before[name].double()
        new = p.detach().double()
        change = new-old
        g = p.grad.detach().double()
        state = optimizer.state[p]
        step = int(state['step'].item())
        rms_adam = (state['exp_avg_sq'].detach().double()/(1-beta2**step)).sqrt()
        old_norm = float(torch.linalg.vector_norm(old).cpu())
        delta_norm = float(torch.linalg.vector_norm(change).cpu())
        result[name] = dict(elements=p.numel(), gradient_l2=float(torch.linalg.vector_norm(g).cpu()),
            gradient_rms=float(g.square().mean().sqrt().cpu()),
            update_l2=delta_norm, update_rms=float(change.square().mean().sqrt().cpu()),
            update_max_abs=float(change.abs().max().cpu()),
            parameter_rms_before=float(old.square().mean().sqrt().cpu()),
            parameter_rms_after=float(new.square().mean().sqrt().cpu()),
            relative_update_l2=delta_norm/old_norm if old_norm else None,
            relative_update_status='defined' if old_norm else 'undefined_zero_initial_parameter',
            changed_coordinates=int(torch.count_nonzero(change).cpu()),
            adam_step=step,
            sqrt_vhat_over_eps_min=float((rms_adam/eps).min().cpu()),
            sqrt_vhat_over_eps_median=float((rms_adam/eps).median().cpu()),
            sqrt_vhat_over_eps_max=float((rms_adam/eps).max().cpu()),
            sqrt_vhat_below_eps_fraction=float((rms_adam < eps).double().mean().cpu()))
    return result


def update(model, current, target, optimizer):
    import torch
    require(all(not v.requires_grad for v in (*current.values(),*target.values())), 'Live encoder input')
    require(all(not p.requires_grad for n,p in model.named_parameters() if not n.startswith('hbf.')), 'Encoder trainable')
    actual = [p for group in optimizer.param_groups for p in group['params']]
    require(len(optimizer.param_groups)==1 and len(actual)==len(list(model.hbf.parameters())) and
            {id(p) for p in actual}=={id(p) for p in model.hbf.parameters()}, 'Unexpected optimizer membership')
    model.hbf.train()
    optimizer.zero_grad(set_to_none=True)
    before = {k:p.detach().clone() for k,p in model.hbf.named_parameters()}
    prediction, delta, h = forward(model, current)
    shared_before = h.clone()
    summary_before, errors_before = summarize(model,current,target,prediction,delta,h)
    losses = model.mo(target,prediction)
    require(torch.isfinite(losses.L_HDEG).item(), 'Nonfinite loss')
    losses.L_HDEG.backward()
    require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.hbf.parameters()),
            'Missing or nonfinite gradient')
    optimizer.step()
    require(all(torch.isfinite(p).all().item() for p in model.hbf.parameters()), 'Nonfinite parameter')
    changes = parameter_updates(model,before,optimizer)
    model.eval()
    with torch.no_grad():
        p,d,h = forward(model,current)
        summary_after, errors_after = summarize(model,current,target,p,d,h)
    return dict(before=summary_before, after=summary_after, parameter_updates=changes,
                shared_feature_change=variation((h-shared_before).cpu().numpy()),
                prediction_change={k:variation((p[k]-prediction[k].detach()).cpu().numpy()) for k in LEVELS},
                losses={k:float(getattr(losses,k).detach().cpu()) for k in ('L_Z','L_S','L_S_tilde','L_G','L_HDEG')}), errors_before, errors_after


def probe(model, current, target, *, step_zero=False):
    import torch
    model.eval()
    # Preserve the original batch size of 32 even for the 64-window probe.
    predictions, deltas, activations = [], [], []
    with torch.no_grad():
        for start in range(0, PROBE_ROWS, 32):
            chunk = {k:v[start:start+32] for k,v in current.items()}
            p,d,h = forward(model,chunk)
            if step_zero:
                require(all(torch.equal(p[k],chunk[k]) and torch.count_nonzero(d[k]).item()==0 for k in LEVELS),
                        'Step zero is not exact persistence')
            predictions.append(p); deltas.append(d); activations.append(h)
        p = {k:torch.cat([v[k] for v in predictions]) for k in LEVELS}
        d = {k:torch.cat([v[k] for v in deltas]) for k in LEVELS}
        return summarize(model,current,target,p,d,torch.cat(activations))
