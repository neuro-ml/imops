import time

import numpy as np
from tqdm import tqdm
from utils import sample_ct, sk_iradon, sk_radon

from imops.radon import inverse_radon, radon


def benchmark(func, *args, **kwargs):
    deltas = []
    value = None
    for _ in tqdm(range(50)):
        start = time.time()
        value = func(*args, **kwargs)
        deltas.append(time.time() - start)

    print('mean', np.mean(deltas), 'std', np.std(deltas))
    return value


image = sample_ct(100, 512)
num_threads = 12

print('Forward radon')
sinogram = benchmark(radon, image, axes=(1, 2), num_threads=num_threads)
benchmark(sk_radon, image)

print('Backward radon')
benchmark(inverse_radon, sinogram, axes=(1, 2), num_threads=num_threads)
benchmark(sk_iradon, sinogram)
