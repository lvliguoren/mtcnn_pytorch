import numpy as np

m = 3
batch_size = 20
if m < batch_size:
    keep_inds = np.arange(m)
    gap = batch_size - m
    while gap >= len(keep_inds):
        gap -= len(keep_inds)
        keep_inds = np.concatenate((keep_inds, keep_inds))
    if gap != 0:
        keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
print(keep_inds)