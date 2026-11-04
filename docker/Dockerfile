# CPU-only image for rebuild_edof.py / edof_align_check.py (e.g. on voodoo under SLURM)
FROM python:3.11-slim
RUN pip install --no-cache-dir "numpy<2.1" opencv-python-headless tifffile h5py imagecodecs
WORKDIR /repo
