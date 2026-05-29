# How to Run AI-Powered Computer-Aided Engineering (CAE) Simulations

This workshop on training AI surrogates for automotive crash dynamics has been presented at GTC 2026.

**Workshop recording:** [DLIT81484](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81484/)

The training pipeline is based on the [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator) crash example and the [GeoTransolver / Transolver](https://github.com/NVIDIA/physicsnemo) models in the PhysicsNeMo GitHub repository.

## Dataset

The crash simulation dataset is hosted on Hugging Face: [AIRBORNEPANDA/BumperBeamCrashExample](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample).

It contains 124 training and 7 validation runs of an OpenRadioss bumper-beam impact simulation, parameterized by impact velocity, shell-element thickness scale, and rigid-wall offset.

Download it before starting the container:

```
pip install huggingface_hub
huggingface-cli download AIRBORNEPANDA/BumperBeamCrashExample \
    --repo-type dataset \
    --local-dir /path/to/data
```

After download, `/path/to/data/` should contain:

- `RAW_DATA/{TRAINING_DATA,VALIDATION_DATA}/Run*/` — OpenRadioss `d3plot` / `*_0000.rad` / `runX.json` files
- `CURATED_DATA_VTP/GLOBAL_FEATURES.json` — per-run design parameters (impact velocity, thickness scale, rigid-wall offset)

Mount this directory to `/data` inside the container (see `docker run -v` below).

## Instructions

- Pull the PhysicsNeMo container from NGC:

```
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.11
```

- Start your container on a system with an NVIDIA GPU:

```
docker run --gpus all --ipc host --pid host --shm-size 16g \
    -v /path/to/GTC2026_AIPoweredCAE/notebooks:/workspace \
    -v /path/to/data:/data \
    -v /path/to/output:/workspace/outputs \
    -p 8888:8888 \
    -p 6006:6006 \
    -t nvcr.io/nvidia/physicsnemo/physicsnemo:25.11
```

- Navigate to `localhost:8888` in a web browser.
- Follow the notebooks in `notebooks/` in order:
    - `00_setup_check.ipynb` — verify the environment, GPU, and dataset are in place.
    - `01_crash_data_processing.ipynb` — run the PhysicsNeMo-Curator ETL pipeline to convert raw OpenRadioss `d3plot` / `*_0000.rad` / `runX.json` files into ML-ready Zarr (and optional VTP) datasets.
    - `02_geotransolver_training.ipynb` — train and evaluate a GeoTransolver surrogate for crash dynamics, including autoregressive rollout, inference, and VTP/GIF visualization.

## Resources

- [Original workshop recording](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81484/)
- [BumperBeamCrashExample dataset on Hugging Face](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample)
- [PhysicsNeMo GitHub repository](https://github.com/NVIDIA/physicsnemo)
- [PhysicsNeMo-Curator GitHub repository](https://github.com/NVIDIA/physicsnemo-curator)
- [OpenRadioss Bumper Beam example](https://openradioss.atlassian.net/wiki/spaces/OPENRADIOSS/pages/11075585/Bumper+Beam)
- [PhysicsNeMo Documentation](https://docs.nvidia.com/physicsnemo)
