# downstreak

This directory contains three main scripts for GEARS-based model training, hyperparameter tuning, and post-training prediction / visualization.

## Scripts Overview

- `gears_model.py`  
  Main script for training a GEARS model and generating evaluation plots.

- `adjust_hyperparameter.py`  
  Script for hyperparameter tuning with Bayesian optimization.

- `predict.py`  
  Script for post-training prediction, GI prediction, and perturbation-level visualization.

---

## 1. gears_model.py

`gears_model.py` is the main training script. It is used to train a GEARS model and generate several evaluation plots after training.

### Main outputs

The main evaluation outputs are saved as:

- `metrics.json`
- `pert_metrics.csv`

### Supported figure formats

Figures can be saved in:

- `svg`
- `png`
- `pdf`

### Supported evaluation plots

#### 1. `fig1_single_and_combo_mse_de`

This figure plots `mse_de` for single-gene and combinational perturbations in the test set.

Other supported metrics include:

- `mse`
- `pearson`
- `pearson_de`

Here, `mse_de` means the mean squared error computed only on differentially expressed (DE) genes rather than on all genes.

#### 2. Per-perturbation summary plots

- `fig2_mse_de_kde_box`  
  KDE + boxplot of `mse_de` across all perturbations.  
  This plot is useful for inspecting the overall distribution, spread, and outliers of DE-gene prediction error.

- `fig2_pearson_de_box`  
  Boxplot of `pearson_de` across perturbations.

- `fig2_scatter_mse_de_vs_pearson_de`  
  Scatter plot showing the relationship between `mse_de` and `pearson_de` for each perturbation.

#### 3. Uncertainty analysis

- `fig3_mse_de_uncertainty_filter_q0.95`  
  Generates uncertainty-analysis plots and summary tables at the perturbation level.

#### 4. GEARS vs No-Perturb comparison

- `fig4_single_and_combo_normalized`  
  Generated when both GEARS and no-perturb runs are available.

  The normalization is defined as:

  `normalized = GEARS / No-Perturb`

  for each subgroup in `metrics.json["single_and_combo"]`.

### Example

```bash
python gears_model.py --save_path test --device cuda --epochs 1 --uncertainty --save_style png
```

---

## 2. adjust_hyperparameter.py

`adjust_hyperparameter.py` is used for hyperparameter tuning with Bayesian optimization.

### Currently supported hyperparameters

- learning rate
- epochs
- weight decay
- hidden size

### Supported modes

#### 1. Single serial Bayesian optimization

This is the traditional Bayesian optimization setting.  
The next trial is selected based on the posterior distribution built from previous trials.

#### 2. Asynchronous Distributed Bayesian Optimization

In this mode, multiple trial points are launched in each round instead of only one.  
All hyperparameter search information is shared through a SQLite database.

This mode is recommended when sufficient computational resources are available.

### Example: single Bayesian optimization

```bash
python adjust_hyperparameter.py --save_path bayesian_single --n_trials 3
```

### Example: parallel Bayesian optimization

```bash
python adjust_hyperparameter.py \
  --study_name para_search_hyperparameters \
  --storage sqlite:////data/projects/31010032/xuanhong/GEARS/downstreak/optuna/gears_hpo.db \
  --save_path bys_para \
  --worker_mode para
```

### PBS scripts for HPC / NSCC

PBS submission scripts are also prepared for NSCC or other HPC environments.

- Run a single Bayesian optimization job on NSCC:

```bash
qsub run_bys_single.sh
```

- Run parallel Bayesian optimization on NSCC:

```bash
qsub run_bys_para.sh
```

---

## 3. predict.py

`predict.py` supports three modes:

1. `predict`
2. `GI_predict`
3. `plot_perturbation`

Before running prediction or plotting, you can first export the available perturbation names and query conditions.

This will generate:

- `pert_name.csv`
- `available_conditions.csv`

under `pred_results/`.

### Export available perturbations and conditions

```bash
python predict.py --export
```

### 3.1 `predict`

This mode is used to predict expression profiles for single-gene or double-gene perturbations.

Main output:

- `predict_result.json`

If the model was trained with uncertainty enabled, it will additionally generate:

- `predict_result_with_uncertainty.json`

#### Examples

Single-gene prediction:

```bash
python predict.py --model_path model/gears_simulation_seed_1/model --pert_list FEV --save_path ./pred_results
```

Double-gene prediction:

```bash
python predict.py --model_path model/gears_simulation_seed_1/model --pert_list FEV+AHR --save_path ./pred_results
```

### 3.2 `GI_predict`

This mode is used to predict genetic interaction (GI) for a two-gene perturbation.

Note:

- exactly two genes are required
- both genes should be valid perturbations supported by the trained model

#### Example

```bash
python predict.py --model_path model/gears_simulation_seed_1/model --combo FEV+AHR --save_path ./pred_results
```

### 3.3 `plot_perturbation`

This mode is used to visualize a comparison between:

- the real expression changes of the top 20 differentially expressed genes
- the predicted expression changes under a given perturbation condition

The generated figure helps inspect how well the model matches the observed response under a specific condition.

#### Example

```bash
python predict.py --model_path model/gears_simulation_seed_1/model --query FEV+ctrl --save_path ./pred_results --save_file ./pred_results
```

---

#### Notes

- `mse_de` refers to the mean squared error on differentially expressed genes.
- `pearson_de` refers to the Pearson correlation on differentially expressed genes.
- `GI_predict` is designed for two-gene perturbation analysis.
- For `plot_perturbation`, the query condition must exist in the available condition list exported from the dataset.

## Code Fixes

1. Fix for uncertainty loss and direction loss

In the original loss function, the uncertainty-based loss and direction loss were handled in the same loop.
The corrected implementation keeps the uncertainty weighting term and direction consistency term explicit for each perturbation subset.

```python
# uncertainty based loss
losses = losses + torch.sum(
    (pred_p - y_p)**(2 + gamma) +
    reg * torch.exp(-logvar_p) * (pred_p - y_p)**(2 + gamma)
) / pred_p.shape[0] / pred_p.shape[1]

# direction loss
if p != 'ctrl':
    losses = losses + torch.sum(
        direction_lambda *
        (torch.sign(y_p - ctrl[retain_idx]) - torch.sign(pred_p - ctrl[retain_idx]))**2
    ) / pred_p.shape[0] / pred_p.shape[1]
else:
    losses = losses + torch.sum(
        direction_lambda *
        (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl))**2
    ) / pred_p.shape[0] / pred_p.shape[1]
```
Notes
-	The uncertainty loss keeps the prediction error term and the variance-weighted regularization term together.
-	The direction loss compares the sign of predicted change and true change relative to the control expression.
-	For non-control perturbations, only the retained indices (retain_idx) are used.
-	For the control case, the full ctrl reference is used directly.

2. Fix for Weights & Biases initialization

The experiment tracking logic was adjusted so that wandb is initialized only when tracking is enabled.
This avoids unnecessary dependency issues and keeps the trainer compatible with runs that do not require logging.

```python
self.weight_bias_track = weight_bias_track
self.config = {}

if self.weight_bias_track:
    import wandb
    wandb.init(
        project=proj_name,
        name=exp_name,
        config=self.config
    )
    self.wandb = wandb
else:
    self.wandb = None
```

Notes
- wandb is imported only when needed.
- self.wandb = None is set when tracking is disabled, so downstream code can safely check whether logging is available.
- configuration dictionary is initialized before wandb.init().