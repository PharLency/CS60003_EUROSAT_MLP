# CS60003_EUROSAT_MLP

Pure NumPy implementation of a three-layer MLP for EuroSAT land-cover classification. This repository includes data loading, backpropagation, training, evaluation, hyperparameter search, and report figure generation for the CS60003 HW1 project.

## Project Structure

```text
.
├── EuroSAT_RGB/              # EuroSAT RGB dataset
├── eurosat_mlp/              # Training / testing / search code
├── report/                   # Report sources and figures
├── hw1.md                    # Assignment description
└── requirements.txt          # Python dependencies
```

## Environment Setup

Python 3.10+ is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you prefer Conda:

```bash
conda create -n eurosat-mlp python=3.10 -y
conda activate eurosat-mlp
pip install -r requirements.txt
```

## Dataset

Place the EuroSAT RGB dataset at:

```text
EuroSAT_RGB/
```

So the class folders should look like:

```text
EuroSAT_RGB/AnnualCrop/
EuroSAT_RGB/Forest/
...
```

## How To Run

All commands below assume you are in the `eurosat_mlp` directory:

```bash
cd eurosat_mlp
```

### 1. Train the main model

```bash
python train.py \
  --data_dir ../EuroSAT_RGB \
  --save_dir checkpoints_aug \
  --hidden1_dim 512 \
  --hidden2_dim 256 \
  --activation relu \
  --lr 0.005 \
  --weight_decay 1e-4 \
  --batch_size 128 \
  --epochs 50 \
  --lr_step_size 15 \
  --lr_gamma 0.5 \
  --augment
```

This will save:

- `checkpoints_aug/best_model.npz`
- `checkpoints_aug/final_model.npz`
- `checkpoints_aug/history.json`

### 2. Evaluate on the test set

```bash
python test.py \
  --data_dir ../EuroSAT_RGB \
  --model_path checkpoints_aug/best_model.npz \
  --save_dir results_aug
```

This will save:

- `results_aug/test_results.json`
- `results_aug/confusion_matrix.npy`
- `results_aug/misclassified.npz`

### 3. Generate training and error-analysis figures

```bash
python visualize.py \
  --history_path checkpoints_aug/history.json \
  --model_path checkpoints_aug/best_model.npz \
  --cm_path results_aug/confusion_matrix.npy \
  --misclassified_path results_aug/misclassified.npz \
  --data_dir ../EuroSAT_RGB \
  --save_dir figures_aug
```

### 4. Run hyperparameter search

Grid search:

```bash
python search.py \
  --data_dir ../EuroSAT_RGB \
  --save_dir search_results_aug \
  --method grid
```

Random search:

```bash
python search.py \
  --data_dir ../EuroSAT_RGB \
  --save_dir search_results_aug \
  --method random \
  --n_trials 20
```

### 5. Generate report figures

```bash
python gen_report_figures.py
```

This writes figures into:

```text
../report/figures/
```

### 6. Run the full pipeline

```bash
python run_all.py
```

By default, this trains, evaluates, and generates figures using the built-in main configuration.

## Notes

- The dataset itself is not included in this repository.
- Large model weights are not tracked in GitHub; upload them separately if needed for submission.
- The report source is available at `report/README.md`.
