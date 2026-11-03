# CPU-only image for the streaming translator.
#
#   docker build -f deploy/Dockerfile -t simulst .
#   docker run --rm -v "$PWD/clip.wav:/audio.wav" simulst --file /audio.wav
#
# The weights are baked in at build time and HF_HUB_OFFLINE is set, so the
# container never reaches the network at run time: no cold-start download, no
# surprise when the hub is slow, and the image is the artefact rather than a
# script that hopes a download succeeds.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    OMP_NUM_THREADS=4

# libsndfile decodes the audio; ffmpeg is only needed for container formats
# soundfile will not open (mp4/m4a). Dropping it saves ~200 MB if the inputs are
# wav, flac or mp3.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first and from the CPU index: the default wheel pulls ~2 GB of CUDA
# libraries that cannot be used here.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# MODEL is a hub id, or a directory inside the build context. The whole deploy/
# directory is copied because a local MODEL path has to exist INSIDE the image
# for fetch_model.py to read it -- copying only the script leaves the weights
# outside and the build fails with a bare exit 1.
#
# Both the staged copy and the pip/HF caches are removed in the same layer, or
# they stay in the image forever at roughly the size of the model.
ARG MODEL=openai/whisper-small
COPY deploy ./deploy
RUN python deploy/fetch_model.py "$MODEL" \
 && rm -rf deploy/model /root/.cache

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY configs ./configs
COPY scripts ./scripts
COPY docs/assets/sample_es.wav ./docs/assets/sample_es.wav

ENTRYPOINT ["python", "-m", "simulst.live", "--device", "cpu", "--checkpoint", "/opt/model"]
CMD ["--file", "docs/assets/sample_es.wav", "--compare"]
