import optuna
import traceback
from gears_model import train_model,evaluate_model,prepare_data
import pandas as pd
import json
import os
import argparse

def sample_hyperparameters(trial):
    """
    used to control the range of hyperparameters.

    parameters:
    ----------
    hidden_size: int
    epochs: int
    lr: float
    weight_decay: float
    """

    params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [4,8,16,32]),
        "epochs": trial.suggest_int("epochs", 10, 40),
        "lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
    }
    return params

def read_metric(save_path, metric_name="test_mse_top20_de_non_dropout"):
    """
    used to get a bayesian optimization target from metric.json.

    parameters:
    -----------
    save_path: str
    metric_name: str

    returns:
    --------
    metric[metric_name]: str

    ### metric_name decide which optimize target to use, can be changed.
    """
    metrics_file = os.path.join(save_path, "metrics.json")
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    if "single_and_combo" in metrics and metric_name in metrics["single_and_combo"]:
        return metrics["single_and_combo"][metric_name]

    if "overall" in metrics and metric_name in metrics["overall"]:
        return metrics["overall"][metric_name]

    if metric_name in metrics:
        return metrics[metric_name]

    raise KeyError(f"{metric_name} not found in {metrics_file}")

def run_single_experiment(
    pert_data,
    params,
    save_path,
    trial_number,
    device="cpu",
    weight_bias_track=False
):
    """
    used to train a single model.
    parameters:
    -----------
    pert_data
    save_path: str
    trial_number: int
    device: str
    weight_bias_track: bool
    returns:
    --------
    save_path: str
    """
    save_path = os.path.join(
        save_path,
        f"trial_{trial_number:04d}_hs{params['hidden_size']}_ep{params['epochs']}_lr{params['lr']:.2e}_wd{params['weight_decay']:.2e}"
    )
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "trial_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    gears_model = train_model(
        pert_data=pert_data,
        save_path=save_path,
        device=device,
        weight_bias_track=weight_bias_track,
        proj_name="gears_hpo",
        exp_name=os.path.basename(save_path),
        hidden_size=params["hidden_size"],
        epochs=params["epochs"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        no_perturb=False,
        uncertainty=False,
    )

    evaluate_model(
        gears_model=gears_model,
        save_test_res=True,
        save_path=save_path
    )

    return save_path


def make_objective(args):
    """
    build a optuna objective.
    """
    pert_data = prepare_data(
        data_path=args.data_path,
        data_name=args.data_name,
        split="simulation",
        seed=args.seed,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        train_gene_set_size=0.75,
        combo_seen2_train_frac=0.75,
        combo_single_split_test_set_fraction=1,
    )

    def objective(trial):
        try:
            params = sample_hyperparameters(trial)
            trial_dir = run_single_experiment(
                pert_data=pert_data,
                params=params,
                save_path=args.save_path,
                device=args.device,
                trial_number=trial.number,
                weight_bias_track=args.weight_bias_track,
            )
            return read_metric(trial_dir, "test_mse_top20_de_non_dropout")
        except Exception as e:
            print(f"[Trial {trial.number}] failed: {e}")
            traceback.print_exc()
            return float("inf")
    return objective

def save_study_results(study, out_csv):
    """
    save study results to a csv file.

    parameters:
    -----------
    study: optuna.study.Study
    out_csv: str
    """
    rows = []
    for t in study.trials:
        row = {
            "trial_number": t.number,
            "value": t.value,
            "state": str(t.state),
        }
        row.update(t.params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def export_study_summary(study_name, storage, save_path):
    """
    export study summary to a csv file from SQLite.
    parameters:
    -----------
    study_name: str
    storage: str
    save_path: str
    """
    os.makedirs(save_path, exist_ok=True)

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )

    save_study_results(
        study,
        os.path.join(save_path, "optuna_trials.csv")
    )
    save_best_params(
        study,
        os.path.join(save_path, "optuna_best_hyperparameter.json")
    )

    print("Best trial:")
    print("  Value:", study.best_value)
    print("  Params:", study.best_params)

def optimize_hyperparameters(
    study_name,
    save_path,
    objective,
    storage=None,
    n_trials=20,
    seed=1,
    work_mode="single"
):
    """
    the main function use to optimize hyperparameters.
    parameters:
    -----------
    study_name: str
    save_path: str
    objective: function
    storage: str
    n_trials: int
    seed: int
    work_mode: str

    return:
    -------
    study: optuna.study.Study
    """
    os.makedirs(save_path, exist_ok=True)

    if work_mode == "para":
        sampler = optuna.samplers.TPESampler()
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler
    )

    try:
        study.optimize(objective, n_trials=n_trials)
    except Exception:
        traceback.print_exc()
        raise
    if work_mode == "single":
        save_study_results(study, os.path.join(save_path, "optuna_trials.csv"))
        save_best_params(study, os.path.join(save_path, "optuna_best_hyperparameter.json"))
    print("Best trial:")
    print("  Value:", study.best_value)
    print("  Params:", study.best_params)

    return study

def save_best_params(study, out_json):
    """
    save study results to a JSON file
    parameters:
    ----------
    study: optuna.study.Study
    out_json: str
    """
    payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_name", type=str,default="norman")
    ap.add_argument("--study_name", type=str,default="hyperparameters")
    ap.add_argument("--storage", type=str)
    ap.add_argument("--save_path", type=str, required=True)
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--weight_bias_track", action="store_true", default=False)
    ap.add_argument("--data_path", type=str, default="./data")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--test_batch_size", type=int, default=128)
    ap.add_argument("--worker_mode", type=str,default="single", choices=["single","para"])
    ap.add_argument("--export_only", action="store_true", default=False)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    ### used to export whole hyperparameters from SQLite
    if args.export_only:
        if args.storage is None:
            raise ValueError("--export_only requires --storage")

        export_study_summary(
            study_name=args.study_name,
            storage=args.storage,
            save_path=args.save_path,
        )
        return

    if args.worker_mode == "para" and args.storage is None:
        raise ValueError("--worker_mode para requires --storage")
    objective = make_objective(args)

    if args.storage is not None:
        if args.worker_mode == "para":
            n_trials = 1
        else:
            n_trials = args.n_trials
        optimize_hyperparameters(
            study_name=args.study_name,
            save_path=args.save_path,
            n_trials=n_trials,
            seed=args.seed,
            storage=args.storage,
            objective=objective,
            work_mode=args.worker_mode,
        )

    else:
        optimize_hyperparameters(
            study_name=args.study_name,
            save_path=args.save_path,
            n_trials=args.n_trials,
            seed=args.seed,
            objective=objective,
            work_mode=args.worker_mode,
        )


if __name__ == "__main__":
    main()