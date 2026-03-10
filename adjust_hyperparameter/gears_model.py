import pickle
import pandas as pd
import io, re
from contextlib import redirect_stdout, redirect_stderr

import seaborn as sns
import torch.nn as nn

import sys

from gears import inference
from gears import *
import numpy as np
import matplotlib.pyplot as plt
import json
import os


### some tools
class TeeStream:
    def __init__(self, original_stream, buffer):
        self.original_stream = original_stream
        self.buffer = buffer

    def write(self, s):
        self.original_stream.write(s)
        self.original_stream.flush()
        self.buffer.write(s)
        return len(s)

    def flush(self):
        self.original_stream.flush()
        self.buffer.flush()


def capture_stdout(func, *args, **kwargs):
    """Run func(*args, **kwargs), keep terminal printing, and also capture stdout+stderr."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    tee_out = TeeStream(sys.stdout, stdout_buf)
    tee_err = TeeStream(sys.stderr, stderr_buf)

    with redirect_stdout(tee_out), redirect_stderr(tee_err):
        ret = func(*args, **kwargs)

    captured = stdout_buf.getvalue() + stderr_buf.getvalue()
    return ret, captured

def parse_test_metrics_from_stdout(text: str) -> dict:
    """Extract 'test_xxx: value' pairs from captured stdout/stderr."""
    # Find occurrences anywhere in a line (prefixes like timestamps are allowed)
    pat = re.compile(r"(test_[A-Za-z0-9_]+)\s*:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
    out = {}
    for line in text.splitlines():
        for m in pat.finditer(line):
            out[m.group(1)] = float(m.group(2))
    return out


# --- New function: parse_best_top20_from_stdout ---
def parse_best_top20_from_stdout(text: str) -> dict:
    """Extract 'Best performing model: Test Top 20 DE MSE: ...' from stdout/stderr."""
    pat = re.compile(
        r"Best\s+performing\s+model\s*:\s*Test\s+Top\s*20\s+DE\s+MSE\s*:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    m = pat.search(text)
    if not m:
        return {}
    return {"test_mse_top20_de_non_dropout": float(m.group(1))}


def compute_uncertainty_per_pert(test_res: dict) -> pd.DataFrame:
    """Compute per-perturbation mean uncertainty from test_res (requires uncertainty=True evaluate).

    Expects:
      - a perturbation label vector under one of: 'pert_cat', 'pert', 'perturbation'
      - an uncertainty matrix under one of: 'logvar', 'unc', 'sigma'

    Returns a DataFrame with columns:
      perturbation, mean_logvar, mean_sigma, certainty
    where certainty = exp(-mean_logvar).
    """
    if not isinstance(test_res, dict):
        raise ValueError("test_res must be a dict to compute uncertainty summaries")

    # locate perturbation labels
    pert_key = None
    for k in ("pert_cat", "pert", "perturbation"):
        if k in test_res:
            pert_key = k
            break
    if pert_key is None:
        raise KeyError("test_res missing perturbation labels (expected one of: pert_cat/pert/perturbation)")

    # locate uncertainty array
    unc_key = None
    for k in ("logvar", "unc", "sigma"):
        if k in test_res:
            unc_key = k
            break
    if unc_key is None:
        raise KeyError("test_res missing uncertainty values (expected one of: logvar/unc/sigma)")

    perts = np.asarray(test_res[pert_key])
    unc = np.asarray(test_res[unc_key])

    # ensure unc is 2D: (n_samples, n_genes)
    if unc.ndim == 1:
        unc = unc.reshape(-1, 1)

    # convert to mean_logvar / mean_sigma consistently
    if unc_key == "logvar":
        mean_logvar_sample = np.mean(unc, axis=1)
        mean_sigma_sample = np.mean(np.exp(0.5 * unc), axis=1)
    elif unc_key == "sigma":
        mean_sigma_sample = np.mean(unc, axis=1)
        mean_logvar_sample = 2.0 * np.log(np.clip(mean_sigma_sample, 1e-12, None))
    else:  # 'unc' (unknown convention) - treat as logvar-like if values look unbounded, else as sigma-like
        # heuristic: if values are frequently negative, likely logvar
        if np.nanmin(unc) < 0:
            mean_logvar_sample = np.mean(unc, axis=1)
            mean_sigma_sample = np.mean(np.exp(0.5 * unc), axis=1)
        else:
            mean_sigma_sample = np.mean(unc, axis=1)
            mean_logvar_sample = 2.0 * np.log(np.clip(mean_sigma_sample, 1e-12, None))

    df = pd.DataFrame({
        "perturbation": perts,
        "mean_logvar_sample": mean_logvar_sample,
        "mean_sigma_sample": mean_sigma_sample,
    })

    g = df.groupby("perturbation", as_index=False).agg({
        "mean_logvar_sample": "mean",
        "mean_sigma_sample": "mean",
    })

    g = g.rename(columns={
        "mean_logvar_sample": "mean_logvar",
        "mean_sigma_sample": "mean_sigma",
    })

    g["certainty"] = np.exp(-g["mean_logvar"].astype(float))
    return g

class TwoOutWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, *args, **kwargs):
        out = self.base(*args, **kwargs)

        if isinstance(out, (tuple, list)):
            if len(out) >= 2:
                p = out[0]
                unc = out[1]
            elif len(out) == 1:
                p = out[0]
                unc = None
            else:
                p = out
                unc = None
        else:
            p = out
            unc = None

        if isinstance(p, (tuple, list)):
            p = p[0] if len(p) > 0 else None
        if isinstance(unc, (tuple, list)):
            unc = unc[0] if len(unc) > 0 else None
        return p, unc

### main functions
def prepare_data(data_path,
                 data_name,
                 split,
                 seed=1,
                 batch_size=32,
                 test_batch_size=128,
                 train_gene_set_size=0.75,
                 combo_seen2_train_frac=0.75,
                 combo_single_split_test_set_fraction=1
                 ):
    pert_data = PertData(data_path)
    pert_data.load(data_name = data_name)
    pert_data.prepare_split(split=split,
                            seed=seed,
                            train_gene_set_size=train_gene_set_size,
                            combo_seen2_train_frac=combo_seen2_train_frac,
                            combo_single_split_test_set_fraction=combo_single_split_test_set_fraction)
    pert_data.get_dataloader(batch_size = batch_size,test_batch_size = test_batch_size)
    print("all the data have been prepared")
    return pert_data


def train_model(pert_data,
                save_path,
                device,
                weight_bias_track=False,
                proj_name="gears_model",
                exp_name="gears_model",
                hidden_size = 64,
                num_go_gnn_layers = 1,
                num_gene_gnn_layers = 1,
                num_similar_genes_go_graph = 16,
                num_similar_genes_co_express_graph = 20,
                coexpress_threshold = 0.4,
                uncertainty = False,
                uncertainty_reg = 1,
                direction_lambda = 1e-1,
                G_go = None,
                G_go_weight = None,
                G_coexpress = None,
                G_coexpress_weight = None,
                no_perturb = False,
                epochs=1,
                lr=0.0001,
                weight_decay = 5e-4
                ):

    effective_uncertainty = bool(uncertainty and not no_perturb)
    if uncertainty and no_perturb:
        print("[WARN] no_perturb mode does not support uncertainty; forcing uncertainty=False for this run.")

    gears_model = GEARS(pert_data = pert_data,
                        device=device,
                        weight_bias_track=weight_bias_track,
                        proj_name=proj_name,
                        exp_name=exp_name,
                        )
    gears_model.model_initialize(hidden_size=hidden_size,
                                 num_go_gnn_layers=num_go_gnn_layers,
                                 num_gene_gnn_layers=num_gene_gnn_layers,
                                 num_similar_genes_go_graph=num_similar_genes_go_graph,
                                 num_similar_genes_co_express_graph=num_similar_genes_co_express_graph,
                                 coexpress_threshold=coexpress_threshold,
                                 uncertainty=effective_uncertainty,
                                 uncertainty_reg=uncertainty_reg,
                                 direction_lambda=direction_lambda,
                                 G_go=G_go,
                                 G_go_weight=G_go_weight,
                                 G_coexpress=G_coexpress,
                                 G_coexpress_weight=G_coexpress_weight,
                                 no_perturb=no_perturb,
                                 )

    if isinstance(getattr(gears_model, "config", None), dict):
        gears_model.config["uncertainty"] = effective_uncertainty
    else:
        gears_model.config = {"uncertainty": effective_uncertainty}

    if effective_uncertainty:
        gears_model.model = TwoOutWrapper(gears_model.model)

    _, train_stdout = capture_stdout(gears_model.train, epochs=epochs, lr=lr, weight_decay=weight_decay)
    subgroup_print_metrics = parse_test_metrics_from_stdout(train_stdout)
    subgroup_print_metrics.update(parse_best_top20_from_stdout(train_stdout))

    # 把打印解析出来的字典挂在对象上，后面 evaluate_model() 写 json 时用
    gears_model._print_metrics = subgroup_print_metrics
    print("model has been trained. saving model now ...")
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        gears_model.save_model(save_path)
    else:
        save_path = os.path.join(os.getcwd(),"model")
        os.makedirs(save_path, exist_ok=True)
        gears_model.save_model(save_path)
    print(f"[OK] saved model to: {save_path}")
    return gears_model

def evaluate_model(gears_model,save_path,save_test_res=True):
    os.makedirs(save_path, exist_ok=True)
    test_loader = gears_model.dataloader['test_loader']
    device = gears_model.device
    cfg = getattr(gears_model, "config", None)
    uncertainty_flag = bool(cfg.get("uncertainty", False)) if isinstance(cfg, dict) else False
    model = gears_model.best_model
    if uncertainty_flag:
        model = TwoOutWrapper(model)
    test_res = inference.evaluate(test_loader, model, uncertainty=uncertainty_flag, device=device)
    test_metrics, test_pert_res = inference.compute_metrics(test_res)

    # --- uncertainty extras (only available when uncertainty=True and test_res contains logvar/unc/sigma) ---
    unc_block = {}
    unc_csv_path = None
    if uncertainty_flag:
        try:
            pert2unc_df = compute_uncertainty_per_pert(test_res)
            unc_csv_path = os.path.join(save_path, "uncertainty_per_pert.csv")
            os.makedirs(save_path, exist_ok=True)
            pert2unc_df.to_csv(unc_csv_path, index=False)

            # attach basic summaries
            unc_block["n_perts"] = int(len(pert2unc_df))
            unc_block["mean_logvar_mean"] = float(np.mean(pert2unc_df["mean_logvar"].values))
            unc_block["mean_sigma_mean"] = float(np.mean(pert2unc_df["mean_sigma"].values))

            # correlation with per-pert metrics (if present)
            tm = pd.DataFrame([
                {"perturbation": k, **{kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float, np.floating, np.integer))}}
                for k, v in test_pert_res.items()
            ])
            merged = pert2unc_df.merge(tm, on="perturbation", how="inner")

            # common metrics to try
            for m in ("mse_de", "mse", "pearson_de", "pearson"):
                if m in merged.columns:
                    x = merged["certainty"].astype(float).values
                    y = merged[m].astype(float).values
                    if len(x) >= 3:
                        # Pearson correlation
                        r = np.corrcoef(x, y)[0, 1]
                        unc_block[f"corr_certainty_vs_{m}"] = float(r)

            unc_block["csv"] = os.path.basename(unc_csv_path)
        except Exception as e:
            # keep pipeline running even if uncertainty keys aren't present
            unc_block["error"] = str(e)

    payload = {
        "overall": test_metrics,
        "single_and_combo": getattr(gears_model, "_print_metrics", {}),
        "uncertainty": unc_block,
    }

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else o)
    print(f"[OK] saved metrics.json to: {save_path}")

    rows = []
    for pert, m in test_pert_res.items():
        row = {"perturbation": pert}
        row.update(m)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(save_path, "pert_metrics.csv"), index=False)
    print(f"[OK] saved pert_metrics.csv to: {save_path}")

    if save_test_res:
        with open(os.path.join(save_path, "test_res.pkl"), "wb") as f:
            pickle.dump(test_res, f)

    return payload

def plot_single_and_combo(save_path, dpi=500,pic_style=None,save_style=None):
    if save_style is None:
        save_style = ["png", "svg", "pdf"]
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "metrics.json"), "r") as f:
        metrics = json.load(f).get("single_and_combo",{})

    if pic_style is None:
        pic_style = ["mse_de", "pearson_de", "mse", "pearson"]
    for p in pic_style:
        rows = []
        for k,v in metrics.items():
            if k.endswith(p):
                name = k.removesuffix(f"_{p}")
                name = name.removeprefix("test_")
                rows.append({
                    "metric_name": name,
                    "value": v
                             })

        df = pd.DataFrame(rows)
        if df.empty:
            print(f"[WARN] No metrics ending with {p} found.")
            continue

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="metric_name", y="value",width=0.6)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("single_and_combo")
        plt.ylabel(p)
        plt.title(f"{p} by Group")
        plt.tight_layout()
        for style in save_style:
            plt.savefig(os.path.join(save_path, f"fig1_single_and_combo_{p}.{style}"), dpi=dpi)
            print(f"[OK] saved fig1_single_and_combo_{p}.{style} to: {save_path}")
        plt.close()

def plot_normalized_top20_mse_de(gears_dir, no_perturb_dir, dpi=500, save_style=None):
    """Plot normalized Top20 DE MSE (non-dropout) for simulation subgroups.

    normalized = GEARS / No-Perturb for each subgroup in metrics.json['single_and_combo'].

    Parameters
    ----------
    gears_dir : str
        Directory containing GEARS run outputs (must contain metrics.json).
    no_perturb_dir : str
        Directory containing No-Perturb run outputs (must contain metrics.json).
    """
    if save_style is None:
        save_style = ["png", "svg", "pdf"]
    save_path = gears_dir
    os.makedirs(save_path, exist_ok=True)

    def _load_top20_df(run_dir: str, label: str) -> pd.DataFrame:
        with open(os.path.join(run_dir, "metrics.json"), "r") as f:
            metrics = json.load(f).get("single_and_combo", {})

        rows = []
        for k_1, v_1 in metrics.items():
            if k_1.endswith("mse_top20_de_non_dropout"):
                name = k_1.removeprefix("test_").removesuffix("_mse_top20_de_non_dropout")
                rows.append({"metric_name": name, "value": v_1, "model": label})

        df_local = pd.DataFrame(rows)
        if df_local.empty:
            print(f"[WARN] No metrics ending with mse_top20_de_non_dropout found in: {run_dir}")
        df_local["value"] = pd.to_numeric(df_local["value"], errors="coerce")
        df_local = df_local.dropna(subset=["value"]).copy()
        return df_local

    df_gears = _load_top20_df(gears_dir, "GEARS")
    df_nop = _load_top20_df(no_perturb_dir, "No-Perturb")

    if df_gears.empty or df_nop.empty:
        print("[WARN] Skip normalized plot because one side is empty.")
        return

    merged = df_gears.merge(df_nop, on="metric_name", how="inner", suffixes=("_gears", "_nop"))
    merged["normalized"] = merged["value_gears"] / merged["value_nop"]

    df_normal = merged[["metric_name", "normalized"]].copy()

    order = ["unseen_single", "combo_seen0", "combo_seen1", "combo_seen2"]
    df_plot = df_normal[df_normal["metric_name"].isin(order)].copy()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_plot, x="metric_name", y="normalized", order=order,width=0.6)
    plt.axhline(1.0, linestyle="dashed")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("single_and_combo")
    plt.ylabel("Normalized Top20 DE MSE")
    plt.tight_layout()

    for style in save_style:
        out_file = os.path.join(save_path, f"fig4_single_and_combo_normalized.{style}")
        plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
        print(f"[OK] saved fig4_single_and_combo_normalized.{style} to: {save_path}")
    plt.close()

def plot_per_perturbation(save_path,save_style,dpi=500,topk=20):
    if save_style is None:
        save_style = ["png", "svg", "pdf"]
    os.makedirs(save_path, exist_ok=True)

    basename = os.path.basename(save_path)
    if basename.startswith("no_perturb"):
        return

    df =  pd.read_csv(os.path.join(save_path,"pert_metrics.csv"))

    for col in ["mse", "mse_de", "pearson_de", "pearson"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df =df.dropna(subset=["mse_de", "pearson_de"]).copy()

    # KDE + boxplot for mse_de
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    sns.kdeplot(data=df, x="mse_de", ax=ax1, fill=False)
    ax1.set_ylabel("Density")
    ax1.set_title("Fig2: Distribution of mse_de (Top 20 DE genes)")

    sns.boxplot(data=df, x="mse_de", ax=ax2, orient="h", showfliers=True,width=0.6)
    ax2.set_xlabel("mse_de")
    ax2.set_yticks([])

    sns.despine()
    fig.tight_layout()

    for style in save_style:
        p = os.path.join(save_path, f"fig2_mse_de_kde_box.{style}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # boxplot for pearson_de
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.boxplot(data=df, x="pearson_de", ax=ax, orient="h", showfliers=True,width=0.3)
    ax.set_title("Fig2: pearson_de across perturbations")
    ax.set_xlabel("pearson_de")
    ax.set_yticks([])
    ax.set_xlim(-1, 1)  # safe default
    sns.despine()
    fig.tight_layout()

    for style in save_style:
        p = os.path.join(save_path, f"fig2_pearson_de_box.{style}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # top K
    K = int(topk)
    top = df.sort_values("mse_de", ascending=False).head(K).copy()

    # Save the table (useful for debugging / reporting)
    top_csv = os.path.join(save_path, f"top{K}_hardest_by_mse_de.csv")
    top.to_csv(top_csv, index=False)

    fig_h = max(4, 0.35 * len(top) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    sns.barplot(data=top, x="mse_de", y="perturbation", ax=ax, orient="h",width=0.6)
    ax.set_title(f"Fig2: Top-{K} hardest perturbations (by mse_de)")
    ax.set_xlabel("mse_de")
    ax.set_ylabel("")
    ax.invert_yaxis()  # largest on top

    sns.despine()
    fig.tight_layout()

    for style in save_style:
        p = os.path.join(save_path, f"fig2_top{K}_hardest_bar.{style}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # scatter mse_de vs pearson_de
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=df, x="mse_de", y="pearson_de", ax=ax, s=25)
    ax.set_title("Fig2: mse_de vs pearson_de (per perturbation)")
    ax.set_xlabel("mse_de")
    ax.set_ylabel("pearson_de")
    ax.set_ylim(-1, 1)

    sns.despine()
    fig.tight_layout()

    for style in save_style:
        p = os.path.join(save_path, f"fig2_scatter_mse_de_vs_pearson_de.{style}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Fig4 outputs saved under: {save_path}")
    print(f"[OK] Top-K table saved: {top_csv}")

def plot_uncertainty(save_path,save_style,dpi=500,quantile=0.95):
    if save_style is None:
        save_style = ["png", "svg", "pdf"]
    os.makedirs(save_path, exist_ok=True)
    df_uncert = pd.read_csv(os.path.join(save_path, "uncertainty_per_pert.csv"))
    df_pert = pd.read_csv(os.path.join(save_path, "pert_metrics.csv"))

    merge = df_uncert.merge(df_pert,
                            on="perturbation",
                            how="inner"
                            )
    df = merge.copy()
    df["mean_logvar"]=pd.to_numeric(df["mean_logvar"], errors="coerce")
    df["mse_de"] = pd.to_numeric(df["mse_de"], errors="coerce")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x="mean_logvar", y="mse_de")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("mean_logvar")
    plt.ylabel("mse_de")
    plt.tight_layout()
    for style in save_style:
        plt.savefig(os.path.join(save_path, f"mean_logvar_with_mse_de.{style}"), dpi=dpi)
        print(f"[OK] saved mean_logvar_with_mse_de.{style} to: {save_path}")
    plt.close()

    #
    thr = float(df["mean_logvar"].quantile(quantile))
    df["unc_group"] = np.where(df["mean_logvar"] >= thr, f"High uncertainty (top {int((1 - quantile) * 100)}%)",
                               "Low uncertainty")

    high = df[df["unc_group"].str.startswith("High")]["mse_de"].values
    low  = df[df["unc_group"] == "Low uncertainty"]["mse_de"].values
    print(f"[INFO] quantile={quantile}, high_unc_threshold(mean_logvar)={thr:.6g}")
    print(f"[INFO] n_low={len(low)}, n_high={len(high)}")

    low_mean, low_med = np.mean(low), np.median(low)
    high_mean, high_med = np.mean(high), np.median(high)

    print(f"[LOW ] mse_de mean={low_mean:.6g}, median={low_med:.6g}")
    print(f"[HIGH] mse_de mean={high_mean:.6g}, median={high_med:.6g}")

    # 提升幅度：过滤掉高不确定（只看 low）相对整体/相对 high
    all_vals = df["mse_de"].values
    all_mean, all_med = np.mean(all_vals), np.median(all_vals)

    print(f"[ALL ] mse_de mean={all_mean:.6g}, median={all_med:.6g}")

    # 相对整体的改善（mse 越小越好，所以是下降比例）
    improve_mean = (all_mean - low_mean) / all_mean
    improve_med = (all_med - low_med) / all_med
    print(f"[IMPROVE vs ALL] mse_de mean ↓ {improve_mean * 100:.2f}%, mse_de median ↓ {improve_med * 100:.2f}%")

    sns.set_theme(style="ticks")
    plt.figure(figsize=(7, 4))
    order = ["Low uncertainty", df["unc_group"].unique()[df["unc_group"].unique() != "Low uncertainty"][0]]
    sns.boxplot(data=df, x="unc_group", y="mse_de", order=order, showfliers=True,width=0.2)
    sns.despine()
    plt.title(f"fig3_mse_de under uncertainty filter")
    plt.xlabel("")
    plt.tight_layout()

    for ext in save_style:
        out_path = os.path.join(save_path, f"fig3_mse_de_uncertainty_filter_q{quantile}.{ext}")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    # Save the merged table for reproducibility
    df_out = df.sort_values(["unc_group", "mse_de"], ascending=True)
    df_out.to_csv(os.path.join(save_path, f"table_mse_de_q{quantile}.csv"), index=False)


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    # prepare data
    ap.add_argument("--data_path",default="./data")
    ap.add_argument("--data_name",default="norman",
                    choices=["norman","adamson","dixit","replogle_k562_essential","replogle_rpe1_essential"])
    ap.add_argument("--split",default="simulation")
    ap.add_argument("--seed",default=1,type=int)
    ap.add_argument("--batch_size",default=100,type=int)
    ap.add_argument("--test_batch_size",default=128,type=int)
    ap.add_argument("--train_gene_set_size",default=0.75,type=float)
    ap.add_argument("--combo_seen2_train_frac",default=0.75,type=float)
    ap.add_argument("--combo_single_split_test_set_fraction",default=0.1,type=float)

    # train module
    ap.add_argument("--save_path",required=True,type=str)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--weight_bias_track", action="store_true", default=False)
    ap.add_argument("--proj_name", default="gears_model", type=str)
    ap.add_argument("--exp_name", default="gears_model", type=str)
    ap.add_argument("--hidden_size", default=64, type=int)
    ap.add_argument("--num_go_gnn_layers", default=1, type=int)
    ap.add_argument("--num_gene_gnn_layers", default=1, type=int)
    ap.add_argument("--num_similar_genes_go_graph", default=16, type=int)
    ap.add_argument("--num_similar_genes_co_express_graph", default=20, type=int)
    ap.add_argument("--uncertainty", action="store_true", default=False)
    ap.add_argument("--uncertainty_reg", default=1, type=int)
    ap.add_argument("--direction_lambda", default=1e-1, type=float)
    ap.add_argument("--no_perturb", action="store_true", default=False)
    ap.add_argument("--epochs",default=20,type=int)
    ap.add_argument("--lr",default=0.0001,type=float)
    ap.add_argument("--weight_decay",default=5e-4,type=float)

    # evaluate model
    ap.add_argument("--save_test_res", action="store_true", default=False)

    # plot model
    ap.add_argument("--save_style", default=["png"], nargs="+", choices=['png','pdf','svg'], type=str)
    ap.add_argument("--pic_style", default=["mse_de"], nargs="+", choices=["mse_de", "pearson_de", "mse", "pearson"], type=str)
    ap.add_argument("--dpi", default=500,type=int)
    ap.add_argument("--topk",default=20,type=int)

    # compare
    ap.add_argument("--compare",action="store_true")
    ap.add_argument("--models", nargs="+", default=["gears", "no_perturb"], choices=["gears", "no_perturb"], help="Which model variants to run")
    return ap.parse_args()

def main():
    args = parse_args()
    pert_data =prepare_data(data_path=args.data_path,
                            data_name=args.data_name,
                            split=args.split,
                            seed=args.seed,
                            batch_size=args.batch_size,
                            test_batch_size=args.test_batch_size,
                            train_gene_set_size=args.train_gene_set_size,
                            combo_seen2_train_frac=args.combo_seen2_train_frac,
                            combo_single_split_test_set_fraction=args.combo_single_split_test_set_fraction
                            )
    run_dirs = {}  # Track per-model output directories
    for model_name in args.models:
        no_perturb_flag = (model_name == 'no_perturb')
        epochs = 0 if no_perturb_flag else args.epochs
        file_name = f"{model_name}_{args.split}_seed_{args.seed}"
        save_path = os.path.join(args.save_path, file_name)
        os.makedirs(save_path, exist_ok=True)
        module_path = os.path.join(save_path, "model")
        os.makedirs(module_path, exist_ok=True)
        run_dirs[model_name] = save_path
        gears_model = train_model(pert_data=pert_data,
                                  save_path=module_path,
                                  device=args.device,
                                  weight_bias_track=args.weight_bias_track,
                                  proj_name=args.proj_name,
                                  exp_name=file_name,
                                  hidden_size=args.hidden_size,
                                  num_go_gnn_layers=args.num_go_gnn_layers,
                                  num_gene_gnn_layers=args.num_gene_gnn_layers,
                                  num_similar_genes_go_graph=args.num_similar_genes_go_graph,
                                  num_similar_genes_co_express_graph=args.num_similar_genes_co_express_graph,
                                  coexpress_threshold=args.coexpress_threshold if hasattr(args, "coexpress_threshold") else 0.4,
                                  uncertainty=args.uncertainty,
                                  uncertainty_reg=args.uncertainty_reg,
                                  direction_lambda=args.direction_lambda,
                                  no_perturb=no_perturb_flag,
                                  epochs=epochs,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay
                                  )
        evaluate_model(gears_model=gears_model,
                       save_test_res=args.save_test_res,
                       save_path=save_path)
        if not no_perturb_flag:
            save_path_store=save_path
            plot_single_and_combo(save_path=save_path, dpi=args.dpi, pic_style=args.pic_style,save_style=args.save_style)
            if args.uncertainty and os.path.exists(os.path.join(save_path, "uncertainty_per_pert.csv")):
                plot_uncertainty(save_path=save_path, save_style=args.save_style, dpi=args.dpi)
            plot_per_perturbation(save_path=save_path, save_style=args.save_style, dpi=args.dpi, topk=args.topk)
        if args.weight_bias_track:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass

    if "gears" in run_dirs and "no_perturb" in run_dirs:
        plot_normalized_top20_mse_de(
            gears_dir=run_dirs["gears"],
            no_perturb_dir=run_dirs["no_perturb"],
            dpi=args.dpi,
            save_style=args.save_style,
        )
    else:
        print("[WARN] normalized plot skipped (need both gears and no_perturb runs)")






if __name__ == "__main__":
    main()