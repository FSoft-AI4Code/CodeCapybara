import numpy as np

def pad_batch_2D(batch, value = 0):
    max_batch = max([len(x) for x in batch])
    batch = [n + [value] * (max_batch - len(n)) for n in batch]
    batch = np.asarray(batch)
    return batch