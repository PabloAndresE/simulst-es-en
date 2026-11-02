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

## Size

The image is several gigabytes, nearly all of it PyTorch and the baked weights.
That is the cost of a container that needs no network: the alternative is a
smaller image that downloads a gigabyte on first request, which fails in front
of whoever you are demonstrating to.

Microphone mode is not available in the container. Audio devices do not cross
the boundary portably, and pretending otherwise would be the kind of claim this
repository is trying to avoid.
