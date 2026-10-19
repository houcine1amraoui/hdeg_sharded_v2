"""Read-only MSE decomposition with coordinate-wise centering over windows."""
import numpy as np
from src.scientific_validation.v2.preflight import require

PREDICTORS = ('original_hbf', 'pilot_hbf', 'persistence', 'train_mean')


def flat(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim >= 2 and len(a) > 0 and np.isfinite(a).all(), 'Invalid representation array')
    return a.reshape(len(a), -1)


def centered(a):
    # Differences from the first row avoid artificial variance for a repeated
    # floating-point constant and improve stability around a nonzero offset.
    delta = a-a[0]
    average_delta = delta.mean(axis=0)
    return a[0]+average_delta, delta-average_delta


def fit_train_mean(training_targets):
    return centered(flat(training_targets))[0]


def decompose(prediction, target):
    p, t = flat(prediction), flat(target)
    require(p.shape == t.shape, 'Prediction/target shapes differ')
    pm, pc = centered(p)
    tm, tc = centered(t)
    per_window = np.mean((p-t)**2, axis=1)
    centered_per_window = np.mean((pc-tc)**2, axis=1)
    vp, vt = np.mean(pc**2, axis=0), np.mean(tc**2, axis=0)
    cov = np.mean(pc*tc, axis=0)
    bias = (pm-tm)**2
    mse, offset, residual = float(per_window.mean()), float(bias.mean()), float(centered_per_window.mean())
    pv, tv, covariance = float(vp.mean()), float(vt.mean()), float(cov.mean())
    expanded = pv+tv-2*covariance
    scale = mse+offset+residual+pv+tv+2*abs(covariance)
    abs_error = max(abs(mse-offset-residual), abs(residual-expanded))
    relative_error = abs_error/scale if scale else 0.0
    require(relative_error <= 1e-9 or abs_error <= 1e-24, 'MSE decomposition identity failed')
    denominator = np.sqrt(pv*tv)
    correlation = float(np.clip(covariance/denominator, -1, 1)) if denominator else None
    summary = dict(windows=len(p), coordinates=p.shape[1], mse=mse,
                   mean_offset_mse=offset, centered_residual_mse=residual,
                   mean_offset_fraction=offset/mse if mse else None,
                   centered_residual_fraction=residual/mse if mse else None,
                   prediction_variance=pv, target_variance=tv, prediction_target_covariance=covariance,
                   centered_alignment=correlation,
                   centered_alignment_status='defined' if correlation is not None else 'undefined_zero_variance',
                   variance_ratio=pv/tv if tv else None,
                   covariance_identity_centered_mse=expanded,
                   decomposition_absolute_residual=abs_error, decomposition_relative_residual=relative_error)
    moments = dict(prediction_mean=pm, target_mean=tm, prediction_variance=vp,
                   target_variance=vt, prediction_target_covariance=cov, mean_offset_squared=bias)
    return summary, per_window, centered_per_window, moments


def compare_to_baseline(errors, baseline):
    p, b = float(np.mean(errors)), float(np.mean(baseline))
    return dict(predictor_mse=p, baseline_mse=b, delta=p-b,
                skill=1-p/b if b else None,
                skill_status='defined' if b else 'undefined_zero_baseline_mse')
