import time
import numpy as np
import minitensor as mt

rng = np.random.default_rng(0)

# (M, K, N)
SHAPES = [
    (100, 2, 100), 
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (1000, 2, 1000), 
    (4096, 4096, 4096),
    (10000, 2, 10000), 
    (10000, 10000, 10000), # don't run this unless you've got some time avalible
]


def time_it(fn, sync=None, warmup=2, min_repeats=10, min_total_s=1.0):
    """Return the median time of fn() in seconds.

    sync: function that waits for the GPU to finish (None for CPU work).
    Repeats until at least min_repeats runs AND min_total_s seconds have passed.
    """
    for _ in range(warmup):          
        fn()
    if sync:
        sync()

    times = []
    while len(times) < min_repeats or sum(times) < min_total_s:
        start = time.perf_counter()
        fn()
        if sync:
            sync()                 
        times.append(time.perf_counter() - start)
        if len(times) >= 100:
            break
    return float(np.median(times))


def gflops(M, K, N, seconds):
    return 2 * M * N * K / seconds / 1e9   


def main():
    header = f"{'M x K x N':>20} | {'naive':>21} | {'tiled':>21} | {'numpy (CPU)':>21}"
    print(header)
    print("-" * len(header))

    for M, K, N in SHAPES:
        a = rng.standard_normal((M, K), dtype=np.float32)
        b = rng.standard_normal((K, N), dtype=np.float32)

        # Copy to the GPU once, outside the timed region
        ta = mt.Tensor.from_numpy(a)
        tb = mt.Tensor.from_numpy(b)

        results = {}
        results["naive"] = time_it(lambda: mt.Tensor.naiveMatMul(ta, tb), sync=mt.synchronize)
        results["tiled"] = time_it(lambda: mt.Tensor.tiledMatMul(ta, tb), sync=mt.synchronize)
        results["numpy"] = time_it(lambda: a @ b)

        cells = [f"{t * 1e3:8.2f} ms {gflops(M, K, N, t):5.0f}GF/s" for t in results.values()]
        print(f"{f'{M} x {K} x {N}':>20} | " + " | ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
