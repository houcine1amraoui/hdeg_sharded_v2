"""Uniform sampling of the original R1 within-shard/chunk pair population."""
import numpy as np


def chunk_population(manifests, chunk_size):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 2:
        raise ValueError('r1-chunk must be an integer >=2')
    chunks = []
    for condition, entry in enumerate(manifests):
        for shard in entry['shards']:
            for start in range(shard['start_index'], shard['end_index'], chunk_size):
                n = min(chunk_size, shard['end_index']-start)
                if n >= 2:
                    chunks.append((condition, shard['shard_index'], start, n, n*(n-1)//2))
    return chunks


def sample_pairs(chunks, capacity, seed):
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError('r1-reservoir must be a positive integer')
    counts = np.array([c[4] for c in chunks], dtype=np.int64)
    ends = np.cumsum(counts)
    if not len(ends) or ends[-1] <= 0:
        raise ValueError('No eligible R1 pairs')
    total = int(ends[-1])
    ranks = np.sort(np.random.default_rng(seed).choice(total, min(total,capacity), replace=False))
    chunk_ids = np.searchsorted(ends, ranks, side='right')
    # Columns: condition index, shard index, global window i, global window j.
    pairs = np.empty((len(ranks),4),dtype=np.int64)
    for cid in np.unique(chunk_ids):
        positions = np.flatnonzero(chunk_ids==cid)
        condition, shard, start, n, _ = chunks[cid]
        local = ranks[positions]-(ends[cid-1] if cid else 0)
        row_starts = np.arange(n,dtype=np.int64)
        row_starts = row_starts*(2*n-row_starts-1)//2
        i = np.searchsorted(row_starts,local,side='right')-1
        j = i+1+local-row_starts[i]
        pairs[positions] = np.column_stack((np.full(len(i),condition),np.full(len(i),shard),start+i,start+j))
    return pairs, ranks, total
