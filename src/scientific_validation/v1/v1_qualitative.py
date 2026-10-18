"""Fixed subset and descriptive pooled PCA for Level 1 visualization."""
import numpy as np

OFFSETS=np.array([0,42,85,127],dtype=np.int64)


def select_windows(plan):
    blocks=plan['primary_blocks']
    if plan['block_length_windows']!=128 or len(blocks)!=32:
        raise ValueError('Expected frozen 32 x 128 primary block plan')
    starts=np.array([b['start_index'] for b in blocks],dtype=np.int64)
    return (starts[:,None]+OFFSETS).reshape(-1)


def pooled_pca(rows):
    x=np.asarray(rows,dtype=np.float64).reshape(len(rows),-1)
    if len(x)<3 or x.shape[1]<2 or not np.isfinite(x).all():
        raise ValueError('PCA requires finite rows and at least two coordinates')
    mean=x.mean(axis=0); centered=x-mean
    u,s,vt=np.linalg.svd(centered,full_matrices=False)
    # Fix arbitrary component signs for repeatable plotting on one numerical stack.
    for j in range(len(s)):
        if vt[j,np.argmax(np.abs(vt[j]))]<0:vt[j]*=-1;u[:,j]*=-1
    energy=s*s;total=energy.sum()
    ratios=energy/total if total>0 else np.zeros_like(energy)
    return dict(scores=u[:,:2]*s[:2],components=vt[:2],mean=mean,
                singular_values=s,explained_variance_ratio=ratios),dict(
        row_count=len(x),coordinate_count=x.shape[1],
        centered_rms_norm=float(np.sqrt(np.mean(np.sum(centered**2,axis=1)))),
        mean_vector_norm=float(np.linalg.norm(mean)),
        pc1_variance_fraction=float(ratios[0]),pc2_variance_fraction=float(ratios[1]),
        zero_centered_variance=bool(total==0),
        interpretation='Centered pooled PCA without per-coordinate scaling. Variance fractions describe this subset; no collapse or semantic-coherence threshold.')
