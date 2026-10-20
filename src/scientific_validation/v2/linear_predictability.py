"""One fixed ridge-linear increment probe; all fitted statistics use fit rows only."""
import numpy as np
from src.scientific_validation.v2.preflight import require
from src.scientific_validation.v2.forecast_diagnostic import LEVELS
from src.scientific_validation.v2.residual_audit import flat, centered, decompose, compare_to_baseline

RIDGE = 0.01
FIT_ROWS = 1024
PREDICTORS = ('linear_increment','mean_increment','persistence')


def partition(indices, blocks, window_size, starts, ends):
    require(len(indices)==len(blocks)==len(starts)==len(ends)==2048, 'Expected frozen 2048 training windows')
    require(np.array_equal(blocks,np.repeat(np.arange(32),64)), 'Unexpected block ordering')
    require(np.all(np.diff(indices)>0) and np.all(np.diff(starts)>0), 'Nonchronological selection')
    require(indices[FIT_ROWS] > indices[FIT_ROWS-1]+window_size and starts[FIT_ROWS]>ends[FIT_ROWS-1],
            'Fit and evaluation raw observation footprints overlap')
    return dict(fit_blocks=list(range(16)),evaluation_blocks=list(range(16,32)),
        fit_rows=FIT_ROWS,evaluation_rows=2048-FIT_ROWS,
        last_fit_target_end_ns=int(ends[FIT_ROWS-1]),first_evaluation_input_start_ns=int(starts[FIT_ROWS]),
        separation_ns=int(starts[FIT_ROWS]-ends[FIT_ROWS-1]),raw_observation_overlap=False)


def fit_transform(current):
    """Per-coordinate centering, one aggregate scale per level; balanced level energy."""
    transformed,stats,details=[],{},{}
    for k in LEVELS:
        a=flat(current[k]);mean,dev=centered(a)
        scale=float(np.sqrt(np.mean(dev*dev)))
        divisor=scale*np.sqrt(a.shape[1]) if scale>0 else 1.0
        x=dev/divisor if scale>0 else np.zeros_like(dev)
        stats[k]=dict(mean=mean,divisor=divisor,active=scale>0)
        details[k]=dict(coordinates=a.shape[1],fit_centered_rms=scale,divisor=divisor,
            active=scale>0,fit_feature_energy=float(np.mean(np.sum(x*x,axis=1))))
        transformed.append(x)
    return np.concatenate(transformed,axis=1),stats,details


def transform(current,stats):
    parts=[]
    for k in LEVELS:
        a=flat(current[k]);s=stats[k]
        require(a.shape[1]==len(s['mean']),'Feature dimension changed')
        parts.append((a-s['mean'])/s['divisor'] if s['active'] else np.zeros_like(a))
    x=np.concatenate(parts,axis=1)
    require(np.isfinite(x).all(),'Nonfinite transformed input')
    return x


def fit_ridge(x,increments):
    """min ||X B - (D-mean(D))||_F^2/n + RIDGE ||B||_F^2; unpenalized intercept."""
    x,y=flat(x),flat(increments)
    require(len(x)==len(y),'Fit row mismatch')
    mean,yc=centered(y)
    n=len(x)
    kernel=x@x.T
    matrix=kernel+n*RIDGE*np.eye(n)
    alpha=np.linalg.solve(matrix,yc)
    require(np.isfinite(alpha).all(),'Nonfinite ridge solution')
    residual=matrix@alpha-yc
    denominator=np.linalg.norm(yc)
    relative=float(np.linalg.norm(residual)/denominator) if denominator else float(np.linalg.norm(residual))
    require(relative<=1e-8,'Ridge linear-system residual too large')
    return dict(x_fit=x,alpha=alpha,mean_increment=mean),dict(
        ridge=RIDGE,fit_rows=n,features=x.shape[1],outputs=y.shape[1],
        dual_system_shape=[n,n],solve_relative_residual=relative,
        objective='sum squared increment errors / n + 0.01 * squared coefficient Frobenius norm',
        intercept='fit-only mean increment, unpenalized')


def predict(model,x):
    values=(flat(x)@model['x_fit'].T)@model['alpha']+model['mean_increment']
    require(np.isfinite(values).all(),'Nonfinite ridge prediction')
    return values


def assess(current,target,linear,mean,block_ids):
    """Errors evaluated in float64 increment space, avoiding addition round-off."""
    output,arrays={},{}
    offset=0
    for k in LEVELS:
        c,t=flat(current[k]),flat(target[k]);increment=t-c;d=c.shape[1]
        predicted=linear[:,offset:offset+d]
        mean_prediction=np.broadcast_to(mean[offset:offset+d],increment.shape)
        values={'linear_increment':predicted,'mean_increment':mean_prediction,'persistence':np.zeros_like(increment)}
        levels={};per={}
        for name,v in values.items():
            summary,errors,_,_=decompose(v,increment)
            levels[name]=summary;per[name]=errors
            arrays[f'{k}__{name}__mse']=errors
        levels['comparisons']={name:compare_to_baseline(per['linear_increment'],per[name])
                               for name in ('persistence','mean_increment')}
        levels['block_comparisons']={}
        for name in ('persistence','mean_increment'):
            diffs=[];rows=[]
            for block in np.unique(block_ids):
                mask=block_ids==block
                a=float(per['linear_increment'][mask].mean());b=float(per[name][mask].mean())
                rows.append(dict(block_id=int(block),linear_mse=a,baseline_mse=b,difference=a-b))
                diffs.append(a-b)
            diffs=np.asarray(diffs)
            levels['block_comparisons'][name]=dict(blocks=rows,improved=int((diffs<0).sum()),
                tied=int((diffs==0).sum()),worsened=int((diffs>0).sum()),
                mean_difference=float(diffs.mean()),median_difference=float(np.median(diffs)))
        levels['mean_increment_vs_persistence']=compare_to_baseline(per['mean_increment'],per['persistence'])
        output[k]=levels;offset+=d
    require(offset==linear.shape[1]==len(mean),'Output coordinate mismatch')
    return output,arrays
