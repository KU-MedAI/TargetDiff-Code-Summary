# TargetDiff Reproduction Scripts

This repository contains a compact rewrite of the core TargetDiff training and sampling workflow used for the `full_test100` reproduction run.

The main reproduced path is:

```text
train_diffusion.py -> checkpoint
sampling.py        -> generated molecules for test set pockets
```

## Files

- `config.yml`: training configuration.
- `sampling.yml`: sampling configuration. By default it uses checkpoint `62000.pt`.
- `train_diffusion.py`: trains the diffusion model.
- `sampling.py`: samples ligands for a selected test-set pocket.
- `dataset.py`, `diffusion.py`, `network.py`, `reconstruct.py`: core model/data/reconstruction code.
- `run_sample_test100.sh`: samples `data_id=0..99` using `62000.pt`.
- `run_full_train_and_sample.sh`: runs training, then samples the test100 pockets from the latest checkpoint.
- `build_lmdb.py`: optional helper for building an LMDB dataset from pocket/ligand files.
- `evaluate_diffusion.py`: auxiliary evaluation script. It depends on TargetDiff-style `utils/` modules and docking tools, so it is not part of the minimal sampling path.

## Required External Artifacts

Large data, checkpoints, and generated results are intentionally not committed. Place these files under the following paths before running the scripts:

```text
data/crossdocked_pocket10_pose_split.pt
data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
logs_diffusion_full/targetdiff_cjkim_full_gpu/checkpoints/62000.pt
```

These paths match `sampling.yml` and `run_sample_test100.sh`.

## Environment

The code expects a Python environment with PyTorch Geometric and chemistry dependencies installed, including:

```text
torch
torch_geometric
torch_scatter
rdkit
openbabel
lmdb
scipy
numpy
pyyaml
tqdm
tensorboard
```

Make sure the PyTorch version matches the installed `torch_scatter` / PyG binary packages.

## Sampling Test100

Run the reproduced test100 sampling job:

```bash
bash run_sample_test100.sh
```

Useful overrides:

```bash
NUM_SAMPLES=100 BATCH_SIZE=16 NUM_STEPS=1000 bash run_sample_test100.sh
```

Outputs are written to:

```text
sampling_results_full_test100/
```

## Training And Sampling

To train and then sample with the latest produced checkpoint:

```bash
bash run_full_train_and_sample.sh
```

Useful overrides:

```bash
TRAIN_MAX_ITERS=71000 TRAIN_TAG=cjkim_full_gpu bash run_full_train_and_sample.sh
```

## Git-Ignored Artifacts

The following are local artifacts and should not be committed:

```text
data/
logs_diffusion*/
sampling_results*/
targetdiff_eval_meta_full_test100/
sampling_runtime*.yml
*.pt
*.lmdb
```
