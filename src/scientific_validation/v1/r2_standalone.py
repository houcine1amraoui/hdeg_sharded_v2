"""Descriptive R2 paired-distance summaries; no independence-based inference."""
import numpy as np
from scipy.stats import rankdata


def correlation_summary(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Expected aligned finite distance vectors')
    result = dict(pair_count=len(x),rho=None,status='undefined',
                  unique_input_distances=len(np.unique(x)),unique_representation_distances=len(np.unique(y)))
    if len(x)<2:
        result['reason']='fewer_than_two_pairs';return result
    if np.all(x==x[0]) or np.all(y==y[0]):
        result['reason']='constant_distance_vector';return result
    a,b=rankdata(x,method='average'),rankdata(y,method='average')
    a-=a.mean();b-=b.mean()
    result.update(rho=float(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)),status='defined')
    return result


def summarize_relationships(x,y,condition_pairs,names=('N1','A2','N2')):
    c=np.asarray(condition_pairs)
    if c.shape != (len(x),2) or not np.issubdtype(c.dtype,np.integer):raise ValueError('Invalid condition pair array')
    if np.any(c<0) or np.any(c>=len(names)) or np.any(c[:,0]>c[:,1]):raise ValueError('Invalid ordered condition indices')
    strata={}
    for a in range(len(names)):
        for b in range(a,len(names)):
            mask=(c[:,0]==a)&(c[:,1]==b)
            strata[f'{names[a]}-{names[b]}']=correlation_summary(x[mask],y[mask])
    return dict(pooled=correlation_summary(x,y),condition_pair_diagnostics=strata,
                diagnostic_scope='Six strata of the same sampled pairs; no separate sampling or significance tests')


def paired_distances(rows, local_pairs, denominator=1, batch_size=1024):
    if denominator<=0:raise ValueError('Invalid normalization')
    result=np.empty(len(local_pairs),dtype=np.float64)
    for start in range(0,len(local_pairs),batch_size):
        stop=min(start+batch_size,len(local_pairs));i,j=local_pairs[start:stop].T
        a=np.asarray(rows[i]).reshape(len(i),-1).astype(np.float64)
        b=np.asarray(rows[j]).reshape(len(j),-1).astype(np.float64)
        result[start:stop]=np.linalg.norm(a-b,axis=1)/denominator
    if not np.isfinite(result).all():raise ValueError('Nonfinite distances')
    return result
