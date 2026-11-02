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

# MODEL can point at a local directory copied into the build context, or at a
# hub id. Either way it is resolved now, not at run time.
ARG MODEL=openai/whisper-small
ENV SIMULST_MODEL=$MODEL
COPY deploy/fetch_model.py ./deploy/fetch_model.py
RUN python deploy/fetch_model.py "$MODEL" && python -c "import shutil; shutil.rmtree('/root/.cache', ignore_errors=True)"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY configs ./configs
COPY scripts ./scripts
COPY docs/assets/sample_es.wav ./docs/assets/sample_es.wav

ENTRYPOINT ["python", "-m", "simulst.live", "--device", "cpu", "--checkpoint", "/opt/model"]
CMD ["--file", "docs/assets/sample_es.wav", "--compare"]
