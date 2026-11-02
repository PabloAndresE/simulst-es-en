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

Measured on this image, `whisper-small`, one 10.4 s utterance, `--chunk-s 1.0`:
see `results/cpu_latency.json`, produced by `scripts/bench_cpu.py`. The number
that matters is per-chunk compute against the chunk budget — above the budget
the policy falls behind the speaker and the latency figures in the README stop
applying.

Microphone mode is not available in the container. Audio devices do not cross
the boundary portably, and pretending otherwise would be the kind of claim this
repository is trying to avoid.
