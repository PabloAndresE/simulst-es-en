# Running it as a container

```bash
# base weights
docker build -f deploy/Dockerfile -t simulst .

# or bake in the fine-tuned checkpoint
cp -r checkpoints/ft-best deploy/model
docker build -f deploy/Dockerfile --build-arg MODEL=deploy/model -t simulst:ft .

docker run --rm simulst                                   # the bundled sample
docker run --rm -v "$PWD/clip.wav:/a.wav" simulst --file /a.wav --compare
```

The weights are resolved during the build and `HF_HUB_OFFLINE=1` is set, so a
started container does no network I/O. That is deliberate: a demo that downloads
1 GB on first request is a demo that fails in front of whoever you are showing it
to.

## What it costs on CPU

`scripts/bench_cpu.py` is in the image, so the measurement can be repeated
wherever the container runs rather than quoted from a different machine:

```bash
docker run --rm --network none --entrypoint python simulst \
    scripts/bench_cpu.py --checkpoint /opt/model --device cpu --chunk-s 1.0 2.0
```

The number that matters is per-chunk compute against the chunk budget. Above the
budget the policy falls behind the speaker and every latency figure in the main
README becomes optimistic by a growing amount.

Measured inside this image on an M1 Pro, fine-tuned `whisper-small`, 8 CPUs
visible to the container:

| Chunk | Median | Max | Budget | Verdict |
|---|---|---|---|---|
| 1.0 s | 2043 ms | 2334 ms | 1000 ms | falls behind |
| 2.0 s | 2309 ms | 2584 ms | 2000 ms | falls behind |
| **3.0 s** | 2199 ms | 2632 ms | 3000 ms | **keeps up** |
| 4.0 s | 2246 ms | 2597 ms | 4000 ms | keeps up |

### Why the container is slower, with the obvious confound ruled out

The image sets `OMP_NUM_THREADS=4` while the container sees 8 CPUs, so the first
suspect was that cap rather than virtualisation. Measured at a one-second chunk,
same model, same audio:

| Threads | Container | Host |
|---|---|---|
| 4 | 2278 ms | 1317 ms |
| 8 | **2735 ms** | 1233 ms |

The cap was not the cause. **More threads make the container slower**, not
faster — virtualised cores contend — so 4 is the right setting here, and at
matched threads the container is still **1.73x slower**.

What remains is where the matrix multiplications run. Whisper's encoder is
almost entirely matmul. On macOS, PyTorch dispatches to Apple's Accelerate
framework, which reaches **AMX**, a dedicated matrix coprocessor on the M-series
silicon. Inside the Linux VM there is no Accelerate and no AMX: the same chip
executes generic ARM kernels on NEON SIMD. Same hardware, no accelerator.

**On real Linux hardware there is no VM** — Docker uses the host kernel directly
— so this gap should mostly disappear. That has **not been measured here** and is
stated as an expectation, not a result.

The practical consequence: **in this container the chunk must be at least 3 s**,
which puts a 3-second floor under latency. That is worse than the 2 s the host
CPU needs and much worse than the sub-2-second figures the main README reports
on a GPU. Compute per chunk is near-constant because the encoder always
processes a padded 30 s window, so a smaller chunk buys responsiveness only
until it stops fitting in the budget.

## Size

The image is several gigabytes, nearly all of it PyTorch and the baked weights.
That is the cost of a container that needs no network: the alternative is a
smaller image that downloads a gigabyte on first request, which fails in front
of whoever you are demonstrating to.

Microphone mode is not available in the container. Audio devices do not cross
the boundary portably, and pretending otherwise would be the kind of claim this
repository is trying to avoid.
