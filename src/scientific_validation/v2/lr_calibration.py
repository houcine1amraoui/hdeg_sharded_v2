"""Fixed train-only learning-rate calibration; no automatic model selection."""
import numpy as np
from src.scientific_validation.v2.preflight import require
from src.scientific_validation.v2.forecast_diagnostic import LEVELS
from src.scientific_validation.v2 import optimization_replay as replay

RATES = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
STEPS = 8
PROBE_AT = (0, 1, 8)


def selections(order, global_indices, window_size):
    require(order.shape == (640,32) and len(global_indices)==2048, 'Unexpected saved selection dimensions')
    updates = order[:8].copy()
    probe = order[8:16].reshape(-1).copy()
    require(len(np.unique(updates))==256 and len(np.unique(probe))==256 and
            len(np.intersect1d(updates,probe))==0, 'Update/probe rows must be distinct')
    # Each paired window covers raw observations [index,index+W], inclusive.
    overlap = np.any(np.abs(global_indices[probe,None]-global_indices[updates.ravel()][None,:]) <= window_size,axis=1)
    return updates, probe, dict(probe_rows_with_raw_overlap=int(overlap.sum()),
        probe_rows_without_raw_overlap=int((~overlap).sum()),
        interpretation='Distinct rows; raw temporal footprints may overlap. Not an independent holdout.')


def reset_candidate(model, initial, rate):
    import torch
    require(rate in RATES, 'Rate outside frozen grid')
    model.hbf.load_state_dict(initial,strict=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    require(all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.hbf.state_dict().items()),
            'Candidate did not reset to identical initialization')
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.hbf.parameters(),lr=rate,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0)
    require(len(optimizer.state)==0,'Optimizer state was not fresh')
    return optimizer


def probe(model,current,target,*,step_zero=False):
    import torch
    require(all(len(v)==256 for v in (*current.values(),*target.values())), 'Expected fixed 256-row probe')
    model.eval()
    predictions,deltas,features=[],[],[]
    with torch.no_grad():
        for start in range(0,256,32):
            chunk={k:v[start:start+32] for k,v in current.items()}
            p,d,h=replay.forward(model,chunk)
            if step_zero:
                require(all(torch.equal(p[k],chunk[k]) and torch.count_nonzero(d[k]).item()==0 for k in LEVELS),
                        'Candidate initialization differs from persistence')
            predictions.append(p);deltas.append(d);features.append(h)
        p={k:torch.cat([v[k] for v in predictions]) for k in LEVELS}
        d={k:torch.cat([v[k] for v in deltas]) for k in LEVELS}
        return replay.summarize(model,current,target,p,d,torch.cat(features))


def comparison_table(curve,updates):
    """Report all candidates; no ranking, acceptance threshold, or automatic choice."""
    result={}
    for label,candidate in curve.items():
        per_level={}
        for k in LEVELS:
            final=candidate['8']['levels'][k]
            after=updates[label][0]['after']['levels'][k]
            before=updates[label][0]['before']['levels'][k]
            def ratio(a,b):return a/b if b else None
            per_level[k]=dict(
                first_update_same_batch_mse_before=before['forecast']['mse'],
                first_update_same_batch_mse_after=after['forecast']['mse'],
                first_update_after_before_ratio=ratio(after['forecast']['mse'],before['forecast']['mse']),
                first_update_correction_increment_energy_ratio=ratio(after['head']['actual_correction']['rms']**2,
                                                                     after['persistence']['mse']),
                final_probe_mse=final['forecast']['mse'],
                probe_persistence_mse=final['persistence']['mse'],
                final_probe_skill=final['comparison']['skill'],
                same_batch_improvements=sum(row['after']['levels'][k]['forecast']['mse'] <
                                            row['before']['levels'][k]['forecast']['mse'] for row in updates[label]),
                final_probe_correction_increment_energy_ratio=ratio(final['head']['actual_correction']['rms']**2,
                                                                    final['persistence']['mse']))
        result[label]=per_level
    return result
