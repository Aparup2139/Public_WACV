# Public_WACV experiments

The `experiments.ipynb` notebook runs QGMamba-CD inference on LEVIR-CD, S2Looking, ValaisCD,
or b-FLAIR. The repository includes the QGMamba-CD weights for LEVIR-CD. A CUDA-capable GPU is
strongly recommended for inference.

## Setup

```bash
git lfs install
git clone https://github.com/Aparup2139/Public_WACV.git
cd Public_WACV
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Install dependencies and start Jupyter:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name public-wacv --display-name "Public WACV"
jupyter lab experiments.ipynb
```

Start Jupyter from the repository root; the notebook uses that directory to resolve its model,
configuration, and checkpoint paths. Select `public-wacv` as the kernel.

## LEVIR-CD notebook inference

### 1. Prepare the dataset

Prepare LEVIR-CD256 with the following layout:

```text
LEVIR-CD256/
|-- A/
|-- B/
|-- label/
`-- list/
    |-- train.txt
    |-- val.txt
    `-- test.txt
```

Each split file must contain the corresponding image filenames, one per line. For example:

```text
train_1.png
train_2.png
```

### 2. Set the dataset path

Open `experiments.ipynb` and set the first code cell as follows, replacing the example path with
the location of LEVIR-CD256 on your machine:

```python
DATASET = "levir_cd"
DATA_ROOT = r"D:\datasets\LEVIR-CD256"
CHECKPOINT_PATH = None
AUTO_DOWNLOAD_DATA = True
SELECT_THRESHOLD_ON_VALIDATION = False
VISUALIZE_SAMPLES = 3
```

Alternatively, leave `DATA_ROOT = None` and define `LEVIR_DATA_ROOT` before starting Jupyter:

```powershell
# Windows PowerShell
$env:LEVIR_DATA_ROOT = "D:\datasets\LEVIR-CD256"
jupyter lab experiments.ipynb
```

```bash
# Linux/macOS
export LEVIR_DATA_ROOT=/path/to/LEVIR-CD256
jupyter lab experiments.ipynb
```

An explicit `DATA_ROOT` value in the notebook takes precedence over the environment variable. If
neither is supplied, the notebook looks for `data/levir_cd` inside the repository.

### 3. Verify the supplied checkpoint

Keep both parts of the supplied LEVIR-CD checkpoint in these repository-relative locations:

```text
Public_WACV/
|-- levir_qgmamba_accuracy_best.pt.zip
`-- best.pt/
    `-- data.pkl
```

Leave `CHECKPOINT_PATH = None`; the LEVIR-CD profile automatically selects the archive and its
matching `config/levir_cd_accuracy_4x40gb.yaml` model configuration.

The supplied checkpoint is a split PyTorch archive. On first use, the notebook combines its ZIP
tensor records with `best.pt/data.pkl` and writes a reusable checkpoint to:

```text
data/.checkpoints/levir_qgmamba_accuracy_best.pt
```

Allow approximately 900 MB of additional disk space for this cached copy. Later runs reuse it.

### 4. Run inference

From the repository root, launch the notebook if it is not already open:

```bash
jupyter lab experiments.ipynb
```

Select **Kernel > Restart Kernel and Run All Cells**. The notebook will:

1. Resolve the LEVIR-CD dataset and checkpoint paths.
2. Build QGMamba-CD with the matching model configuration.
3. Load the trained weights.
4. Index the training, validation, and test splits.
5. Run inference on the held-out test split.
6. Print evaluation metrics and display sample predictions.
7. Report model-efficiency statistics.

By default, `SELECT_THRESHOLD_ON_VALIDATION = False` uses the operating threshold stored in the
checkpoint. Set it to `True` to select a threshold on the validation split before test inference.
`VISUALIZE_SAMPLES` controls the number of qualitative predictions displayed. `CHECKPOINT_PATH`
may be set to another architecture-compatible checkpoint when needed.

If the notebook reports `Start Jupyter from the cloned Public_WACV repository root`, stop the
Jupyter server, change to the cloned `Public_WACV` directory, and launch it again.

## Other datasets

Set `DATASET` in the first code cell to `s2looking`, `valais_bmd`, or `b_flair`. Their dataset
files are downloaded from Hugging Face on first use and cached under `data/`; downloads are
resumable. Approximate source download sizes are 11 GB for S2Looking, 5.5 GB for ValaisCD, and
5 GB for b-FLAIR.

## Checkpoints

Inference requires trained weights and never falls back to random initialization. LEVIR-CD uses
the bundled checkpoint described above. Each other dataset's `config.yaml` first looks for its
`checkpoint_path` inside the clone. If it is absent, the notebook downloads
`checkpoint_hf_filename` from `checkpoint_hf_repo` when a repository is configured.

Expected repository paths are:

- `levir_qgmamba_accuracy_best.pt.zip` plus `best.pt/data.pkl` for LEVIR-CD inference.
- `s2looking/qgmamba_s2looking_best.pt` for S2Looking and b-FLAIR zero-shot inference.
- `valais_bmd/checkpoints/best.pt` for ValaisCD inference.
