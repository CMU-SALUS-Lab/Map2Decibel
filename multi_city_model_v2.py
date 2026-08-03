"""
multi_city_model.py

PURPOSE:
  Evaluate whether a unified noise proxy model can generalise across cities.
  Compares Random Forest vs XGBoost in all experiments.

THREE EXPERIMENTS:
  1. Cross-city transfer     -- saved city model predicts another city
  2. Leave-one-city-out      -- train on N-1 cities, test on held-out city
  3. Unified pooled model    -- train once on all cities, save for deployment

USAGE:
  python multi_city_model.py
  python multi_city_model.py noise_contours_*.gpkg

REQUIREMENTS:
  pip install joblib scikit-learn xgboost geopandas pandas numpy matplotlib
"""

import os
import re
import sys
import glob
import time
import argparse
import joblib
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import QuantileTransformer

try:
    from xgboost import XGBRegressor
    _XGB = True
except ImportError:
    _XGB = False
    print("Note: xgboost not installed. Run: pip install xgboost")

try:
    import torch
    import torch_geometric
    _GNN_AVAILABLE = True
except ImportError:
    _GNN_AVAILABLE = False

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
MODELS_DIR    = "models"
OUTPUT_DIR    = "."
NOISE_DB_COL  = "Lden"
BUFFER_DIST_M = 25
GNN_OPTION    = False   # set True here OR pass --gnn flag
# (resolved after _parse_args() below)

DEFAULT_NOISE_FILES = sorted(glob.glob("noise_contours_*.gpkg"))

# ── Argument parsing ─────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(add_help=True,
        description="Multi-city noise proxy model training and evaluation")
    p.add_argument("files", nargs="*",
                   help="Noise contour .gpkg files (default: noise_contours_*.gpkg)")
    p.add_argument("--gnn", action="store_true",
                   help="Include GIN graph neural network in pooled model comparison")
    p.add_argument("--no-exp1", action="store_true",
                   help="Skip Experiment 1 (cross-city transfer matrix)")
    p.add_argument("--no-exp2", action="store_true",
                   help="Skip Experiment 2 (LOOCV)")
    p.add_argument("--no-exp3", action="store_true",
                   help="Skip Experiment 3 (pooled unified model)")
    args, _ = p.parse_known_args()
    return args

_ARGS       = _parse_args()
GNN_OPTION  = GNN_OPTION or _ARGS.gnn

CITY_NAMES = {
    "pittsburgh": "Pittsburgh, Pennsylvania, USA",
    "amsterdam":  "Amsterdam, Netherlands",
    "zurich":     "Zurich, Switzerland",
    "singapore":  "Singapore",
    "bangkok":    "Bangkok, Thailand",
    "oslo":       "Oslo, Norway",
    "london":     "London, United Kingdom",
    "tokyo":      "Tokyo, Japan",
    "newyork":    "New York City, New York, USA",
    "chicago":    "Chicago, Illinois, USA",
    "sydney":     "Sydney, New South Wales, Australia",
}

# Import canonical feature list from shared module
# This ensures multi_city_model always uses the same features as the pipeline
from noisy_feature_extraction_v1 import FEATURE_COLS, CITY_NAMES

# =============================================================================
# HELPERS
# =============================================================================
def slug_from_file(noise_file):
    return re.sub(r"^noise_contours_", "",
                  os.path.splitext(os.path.basename(noise_file))[0])


def resolve_city_name(slug):
    if slug in CITY_NAMES:
        return CITY_NAMES[slug]
    candidate = slug.replace("_", " ").title()
    try:
        import osmnx as _ox
        gdf = _ox.geocode_to_gdf(candidate)
        if len(gdf) > 0:
            CITY_NAMES[slug] = candidate
            return candidate
    except Exception:
        pass
    CITY_NAMES[slug] = slug
    return slug


def _zv(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def _zn(arr):
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0) + 1e-9
    return (arr - mu) / sd


def _qt(arr_fit, arr_transform):
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    qt.fit(arr_fit)
    return qt.transform(arr_transform)


def make_models():
    """Return dict of candidate models."""
    models = {
        "rf": RandomForestRegressor(
            n_estimators=300, max_features="sqrt",
            min_samples_leaf=5, random_state=42, n_jobs=-1),
    }
    if _XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0)
    return models


# =============================================================================
# LOAD CITY DATA
# =============================================================================
def load_city(noise_file):
    slug  = slug_from_file(noise_file)
    cache = os.path.join(OUTPUT_DIR, "features_cache_{}.gpkg".format(slug))

    if not os.path.exists(cache):
        print("  ERROR: cache not found for {} — run noise_proxy_pipeline_v5.py".format(slug))
        return None, None, slug

    print("  Loading {}...".format(slug))
    em  = gpd.read_file(cache)
    crs = em.crs.to_string()
    print("    {} segments from cache".format(len(em)))

    # Load and spatial join noise contours
    noise = gpd.read_file(noise_file)

    def to_midpoint(v):
        v = str(v).strip()
        if "-" in v:
            a, b = v.split("-"); return (float(a) + float(b)) / 2
        elif ">" in v:
            return float(v.replace(">","").strip()) + 2.5
        try: return float(v)
        except: return np.nan

    db_col = NOISE_DB_COL
    for col in [NOISE_DB_COL, "Lden", "lden", "db", "level"]:
        if col in noise.columns:
            db_col = col; break

    noise["db_mid"] = noise[db_col].apply(to_midpoint)
    noise = noise[["geometry","db_mid"]].dropna().to_crs(crs)
    em    = em.to_crs(crs).reset_index(drop=True)

    edges_buf = gpd.GeoDataFrame(
        {"pos": np.arange(len(em)),
         "geometry": em.geometry.buffer(BUFFER_DIST_M)}, crs=crs)
    joined = gpd.sjoin(edges_buf, noise[["geometry","db_mid"]],
                       how="left", predicate="intersects")
    joined = joined.dropna(subset=["db_mid"])
    noise_dict = joined.groupby("pos")["db_mid"].mean().to_dict()
    em["noise_db"] = em.index.map(noise_dict)

    feat_cols = [f for f in FEATURE_COLS if f in em.columns]
    missing   = [f for f in FEATURE_COLS if f not in em.columns]
    if missing:
        print("    WARNING: {} features missing: {}".format(len(missing), missing))

    df = em[feat_cols + ["noise_db"]].dropna()
    print("    {} labelled segments | {} features".format(len(df), len(feat_cols)))
    return df, feat_cols, slug


def load_saved_model(slug, model_type="rf"):
    path = os.path.join(MODELS_DIR, "{}_{}.joblib".format(slug, model_type))
    if not os.path.exists(path):
        return None, None
    bundle = joblib.load(path)
    return bundle["model"], bundle.get("feature_cols", FEATURE_COLS)


# =============================================================================
# EXPERIMENT 1: CROSS-CITY TRANSFER USING SAVED MODELS
# =============================================================================
def experiment_transfer(city_datasets):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Cross-City Transfer Using Saved Models")
    print("=" * 70)

    slugs    = list(city_datasets.keys())
    feat_ref = FEATURE_COLS

    model_types = ["rf"] + (["xgb"] if _XGB else [])
    all_results  = {}

    for mtype in model_types:
        results_raw  = pd.DataFrame(index=slugs, columns=slugs, dtype=float)
        results_norm = pd.DataFrame(index=slugs, columns=slugs, dtype=float)
        results_mae  = pd.DataFrame(index=slugs, columns=slugs, dtype=float)

        for src_slug in slugs:
            model, _ = load_saved_model(src_slug, mtype)
            if model is None:
                print("  No saved {} model for {}".format(mtype, src_slug))
                continue
            df_src = city_datasets[src_slug][0]
            common = [f for f in feat_ref if f in df_src.columns]
            X_src  = df_src[common].values
            y_src  = df_src["noise_db"].values

            for tgt_slug in slugs:
                df_tgt = city_datasets[tgt_slug][0]
                X_tgt  = df_tgt[common].values
                y_tgt  = df_tgt["noise_db"].values
                try:
                    y_pred = model.predict(X_tgt)
                    results_raw.loc[src_slug, tgt_slug]  = r2_score(y_tgt, y_pred)
                    results_mae.loc[src_slug, tgt_slug]  = mean_absolute_error(y_tgt, y_pred)

                    # Normalised
                    Xs_z = _zn(X_src); ys_z = _zv(y_src)
                    Xt_z = _zn(X_tgt); yt_z = _zv(y_tgt)
                    m_z  = make_models()[mtype]
                    m_z.fit(Xs_z, ys_z)
                    results_norm.loc[src_slug, tgt_slug] = r2_score(yt_z, m_z.predict(Xt_z))
                except Exception as e:
                    print("  Error {}->{} ({}): {}".format(src_slug, tgt_slug, mtype, e))

        all_results[mtype] = (results_raw, results_norm, results_mae)

        print("\n  [{}] Raw R2 matrix:".format(mtype.upper()))
        print("  " + results_raw.round(3).to_string())
        print("\n  [{}] Raw MAE matrix (dB):".format(mtype.upper()))
        print("  " + results_mae.round(2).to_string())
        print("\n  [{}] Normalised R2 matrix:".format(mtype.upper()))
        print("  " + results_norm.round(3).to_string())

        results_raw.to_csv(os.path.abspath("transfer_raw_{}.csv".format(mtype)))
        results_norm.to_csv(os.path.abspath("transfer_norm_{}.csv".format(mtype)))

    print("\n  Saved -> transfer_raw_*.csv, transfer_norm_*.csv")
    return all_results


# =============================================================================
# EXPERIMENT 2: LEAVE-ONE-CITY-OUT
# =============================================================================
def experiment_loocv(city_datasets):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Leave-One-City-Out Cross-Validation")
    print("=" * 70)

    slugs   = list(city_datasets.keys())
    common  = FEATURE_COLS
    for slug in slugs:
        df, feats, _ = city_datasets[slug]
        common = [f for f in common if f in df.columns]

    models      = make_models()
    model_names = list(models.keys())

    # Header
    r2_hdr = "  ".join("{:>9}".format(n.upper()+" R2") for n in model_names)
    print("\n  +-- Leave-One-City-Out (z-normalised pooling) " + "-"*24)
    print("  |  {:<14}  {}  {:>9}  {:>9}  {:>7}".format(
        "Test city", r2_hdr, "MAE(dB)", "NormMAE", "n_test"))
    print("  |  " + "-"*(60 + 11*(len(model_names)-1)))

    results  = {}
    t_loocv  = {}
    for test_slug in slugs:
        t0 = time.perf_counter()
        train_slugs = [s for s in slugs if s != test_slug]
        df_te = city_datasets[test_slug][0]

        X_te     = df_te[common].values
        y_te     = df_te["noise_db"].values
        X_tr_raw = np.vstack([city_datasets[s][0][common].values for s in train_slugs])
        y_tr_raw = np.concatenate([city_datasets[s][0]["noise_db"].values for s in train_slugs])
        X_tr_z   = np.vstack([_zn(city_datasets[s][0][common].values) for s in train_slugs])
        y_tr_z   = np.concatenate([_zv(city_datasets[s][0]["noise_db"].values) for s in train_slugs])
        X_te_z   = _zn(X_te)
        y_te_z   = _zv(y_te)

        # Raw R2 with RF
        rf_raw = RandomForestRegressor(
            n_estimators=300, max_features="sqrt",
            min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf_raw.fit(X_tr_raw, y_tr_raw)
        r2_raw  = r2_score(y_te, rf_raw.predict(X_te))
        mae_raw = mean_absolute_error(y_te, rf_raw.predict(X_te))

        # Normalised R2 per model
        norm_r2s   = {}
        norm_maes  = {}
        for mname, mmodel in models.items():
            mmodel.fit(X_tr_z, y_tr_z)
            y_pred_z = mmodel.predict(X_te_z)
            norm_r2s[mname]  = r2_score(y_te_z, y_pred_z)
            norm_maes[mname] = mean_absolute_error(y_te_z, y_pred_z) * y_te.std()

        # Rank R2
        X_tr_q = np.vstack([_qt(city_datasets[s][0][common].values,
                                 city_datasets[s][0][common].values)
                             for s in train_slugs])
        X_te_q = _qt(X_tr_raw, X_te)
        rf_raw.fit(X_tr_q, y_tr_z)
        r2_rank = r2_score(y_te_z, rf_raw.predict(X_te_q))

        results[test_slug] = {
            "r2_raw":    r2_raw,
            "r2_rank":   r2_rank,
            "mae_raw_db": mae_raw,
            "n_test":    len(y_te),
            **{"r2_norm_{}".format(k): v for k, v in norm_r2s.items()},
            **{"mae_norm_{}".format(k): v for k, v in norm_maes.items()},
        }
        results[test_slug]["r2_norm"] = norm_r2s.get("rf", 0)

        t_loocv[test_slug] = time.perf_counter() - t0
        r2_str = "  ".join("{:>9.3f}".format(norm_r2s.get(n, np.nan))
                           for n in model_names)
        print("  |  {:<14}  {}  {:>9.2f}  {:>9.2f}  {:>7}  {:>6.1f}s".format(
            test_slug, r2_str, mae_raw,
            norm_maes.get("rf", 0), len(y_te),
            t_loocv[test_slug]))

    # Means
    print("  |  " + "-"*(60 + 11*(len(model_names)-1)))
    r2_means = {n: np.mean([v.get("r2_norm_{}".format(n), np.nan)
                             for v in results.values()])
                for n in model_names}
    r2_mean_str = "  ".join("{:>9.3f}".format(r2_means[n]) for n in model_names)
    mae_mean = np.mean([v["mae_raw_db"] for v in results.values()])
    norm_mae_mean = np.mean([v.get("mae_norm_rf", 0) for v in results.values()])
    print("  |  {:<14}  {}  {:>9.2f}  {:>9.2f}".format(
        "MEAN", r2_mean_str, mae_mean, norm_mae_mean))
    print("  +" + "-"*(62 + 11*(len(model_names)-1)))
    print()
    print("  Interpretation:")
    print("  Raw R2 > 0      -> pooled model generalises without normalisation")
    print("  Norm R2 > 0     -> relative noise patterns transfer across cities")
    print("  Norm R2 > 0.3   -> unified model viable for MWI pipeline")
    if _XGB:
        best = max(r2_means, key=r2_means.get)
        print("  Best model: {} (mean norm R2={:.3f})".format(
            best.upper(), r2_means[best]))

    out = os.path.abspath("loocv_results.csv")
    pd.DataFrame(results).T.to_csv(out)
    print("  Results saved -> {}".format(out))
    return results


# =============================================================================
# EXPERIMENT 3: POOLED UNIFIED MODEL
# =============================================================================
def experiment_pooled(city_datasets):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Pooled Unified Model — RF vs XGBoost")
    print("=" * 70)

    slugs  = list(city_datasets.keys())
    common = FEATURE_COLS
    for slug in slugs:
        df, feats, _ = city_datasets[slug]
        common = [f for f in common if f in df.columns]

    print("  Training on {} cities | {} features".format(len(slugs), len(common)))
    print("  Cities: {}".format(", ".join(slugs)))

    X_all_z = np.vstack([_zn(city_datasets[s][0][common].values) for s in slugs])
    y_all_z = np.concatenate([_zv(city_datasets[s][0]["noise_db"].values) for s in slugs])
    X_all   = np.vstack([city_datasets[s][0][common].values for s in slugs])
    y_all   = np.concatenate([city_datasets[s][0]["noise_db"].values for s in slugs])

    models = make_models()
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n  +-- Pooled Model CV (z-normalised) ---------------------------")
    print("  |  {:<8}  {:>8}  {:>8}  {:>10}  {:>9}".format(
        "Model", "R2 mean", "R2 std", "MAE (std)", "Time (s)"))
    print("  |  " + "-"*48)

    cv_results = {}
    for name, model in models.items():
        t0  = time.perf_counter()
        r2  = cross_val_score(model, X_all_z, y_all_z, cv=kf, scoring="r2")
        mae = cross_val_score(model, X_all_z, y_all_z, cv=kf,
                              scoring="neg_mean_absolute_error")
        elapsed = time.perf_counter() - t0
        cv_results[name] = {"r2_mean": r2.mean(), "r2_std": r2.std(),
                            "mae_mean": -mae.mean(), "time_s": elapsed}
        print("  |  {:<8}  {:>8.3f}  {:>8.3f}  {:>10.3f}  {:>9.1f}".format(
            name.upper(), r2.mean(), r2.std(), -mae.mean(), elapsed))

    # Optional GNN in pooled experiment
    if GNN_OPTION:
        if not _GNN_AVAILABLE:
            print("  |  GIN      -- install torch + torch_geometric to enable")
        else:
            print("  |  GIN      training...", flush=True)
            # Import GNN helpers from pipeline
            try:
                import importlib.util, sys as _sys
                _spec = importlib.util.spec_from_file_location(
                    "pipeline", os.path.join(OUTPUT_DIR, "noise_proxy_pipeline_v5.py"))
                if _spec:
                    _pipe = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_pipe)
                    t0 = time.perf_counter()
                    # Use a small sample for speed in multi-city context
                    n_sample = min(len(X_all_z), 10000)
                    idx = np.random.choice(len(X_all_z), n_sample, replace=False)
                    # Create dummy df with u/v = None (MLP fallback)
                    _df_dummy = pd.DataFrame(X_all_z[idx], columns=common)
                    gin_r2, gin_r2std, gin_mae = _pipe.evaluate_gin_cv(
                        X_all_z[idx], y_all_z[idx], _df_dummy)
                    elapsed = time.perf_counter() - t0
                    cv_results["gin"] = {"r2_mean": gin_r2, "r2_std": gin_r2std,
                                         "mae_mean": gin_mae, "time_s": elapsed}
                    print("  |  {:<8}  {:>8.3f}  {:>8.3f}  {:>10.3f}  {:>9.1f}".format(
                        "GIN", gin_r2, gin_r2std, gin_mae, elapsed))
            except Exception as e:
                print("  |  GIN      error: {} -- run --gnn in pipeline script instead".format(e))

    print("  +" + "-"*50)
    best_name = max({k: v for k, v in cv_results.items() if k != "gin"},
                    key=lambda k: cv_results[k]["r2_mean"])
    print("  Best: {} (R2={:.3f})".format(
        best_name.upper(), cv_results[best_name]["r2_mean"]))

    # Timing summary
    print("\n  +-- Training time (pooled model CV) " + "-"*20)
    t_max = max(v["time_s"] for v in cv_results.values()) or 1
    for name, res in sorted(cv_results.items(), key=lambda x: x[1]["time_s"]):
        bar = chr(9608) * int(res["time_s"] / t_max * 20)
        print("  |  {:<8}  {:>7.1f}s  {}".format(
            name.upper(), res["time_s"], bar))
    print("  +" + "-"*38)

    # Fit all on full data and save
    os.makedirs(MODELS_DIR, exist_ok=True)
    fitted = {}
    for name, model in models.items():
        model.fit(X_all_z, y_all_z)
        fitted[name] = model
        path = os.path.join(MODELS_DIR, "unified_norm_{}.joblib".format(name))
        joblib.dump({"model": model, "feature_cols": common,
                     "cities": slugs, "normalised": True,
                     "model_type": name,
                     "cv_r2": cv_results[name]["r2_mean"]}, path)
        print("  Saved -> {}".format(path))

    # Save raw RF for backwards compatibility
    rf_raw = RandomForestRegressor(
        n_estimators=300, max_features="sqrt",
        min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_raw.fit(X_all, y_all)
    joblib.dump({"model": rf_raw, "feature_cols": common,
                 "cities": slugs, "normalised": False},
                os.path.join(MODELS_DIR, "unified_raw_rf.joblib"))

    # Best model becomes default for predict_new_city.py
    best_model = fitted[best_name]
    joblib.dump({"model": best_model, "feature_cols": common,
                 "cities": slugs, "normalised": True,
                 "model_type": best_name,
                 "note": "Best unified model — used by predict_new_city.py"},
                os.path.join(MODELS_DIR, "unified_norm_rf.joblib"))
    print("  Default model (unified_norm_rf.joblib) = {}".format(best_name.upper()))

    # Feature importances
    for name, model in fitted.items():
        if hasattr(model, "feature_importances_"):
            imps = pd.Series(
                model.feature_importances_, index=common
            ).sort_values(ascending=False)
            print("\n  Feature importances ({} unified z-norm):".format(name.upper()))
            for feat, imp in imps.items():
                bar = "#" * int(imp * 40)
                print("    {:<32} {:.3f}  {}".format(feat, imp, bar))

    return fitted, cv_results, common


# =============================================================================
# VISUALISATION
# =============================================================================
def plot_loocv(loocv_results, city_datasets):
    slugs = list(loocv_results.keys())
    model_names = [k.replace("r2_norm_","") for k in loocv_results[slugs[0]]
                   if k.startswith("r2_norm_")]
    if not model_names:
        model_names = ["rf"]

    # Load within-city R2 from saved model results
    within_r2 = {}
    for slug in slugs:
        path = os.path.join(MODELS_DIR, "{}_results.csv".format(slug))
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            if "r2_mean" in df.columns and "rf" in df.index:
                within_r2[slug] = df.loc["rf", "r2_mean"]

    n_bars = 1 + len(model_names) + 1  # within + norm models + rank
    w      = 0.7 / n_bars
    x      = np.arange(len(slugs))
    colors = ["#028090", "#e05c00", "#c05000", "#7fb800", "#5c6bc0"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Within-city bars
    r2_w = [within_r2.get(s, 0) for s in slugs]
    ax.bar(x - (n_bars/2 - 0.5)*w, r2_w, w,
           label="Within-city RF", color=colors[0], alpha=0.9)

    # LOOCV per model
    for i, mname in enumerate(model_names):
        r2_m = [loocv_results[s].get("r2_norm_{}".format(mname), 0) for s in slugs]
        ax.bar(x - (n_bars/2 - 1.5 - i)*w, r2_m, w,
               label="LOOCV {}".format(mname.upper()),
               color=colors[i+1], alpha=0.85)

    # Rank bars
    r2_rank = [loocv_results[s].get("r2_rank", 0) for s in slugs]
    ax.bar(x + (n_bars/2 - 0.5)*w, r2_rank, w,
           label="LOOCV Rank", color=colors[-1], alpha=0.75)

    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(slugs, fontsize=10, color="white")
    ax.set_ylabel("R2", color="white", fontsize=11)
    ax.set_title(
        "Within-city vs Leave-One-City-Out R2\nOSM Morphology -> CNOSSOS Noise",
        color="white", fontsize=12)
    ax.set_ylim(-0.5, 1.05)
    leg = ax.legend(fontsize=9, labelcolor="white",
                    facecolor="#1e2230", edgecolor="none")

    plt.tight_layout()
    out = os.path.abspath("loocv_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  Plot saved -> {}".format(out))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    noise_files = _ARGS.files if _ARGS.files else DEFAULT_NOISE_FILES

    if not noise_files:
        print("No noise contour files found.")
        print("Usage: python multi_city_model.py noise_contours_*.gpkg")
        return

    print("Found {} noise files: {}".format(
        len(noise_files), [os.path.basename(f) for f in noise_files]))
    if _XGB:
        print("XGBoost available — will compare RF vs XGB in all experiments")
    else:
        print("XGBoost not available — RF only (pip install xgboost)")

    print("\nLoading city datasets...")
    city_datasets = {}
    for nf in noise_files:
        df, feats, slug = load_city(nf)
        if df is not None and len(df) > 0:
            city_datasets[slug] = (df, feats, slug)

    if len(city_datasets) < 2:
        print("Need at least 2 cities with data.")
        return

    print("\nLoaded {} cities: {}".format(
        len(city_datasets), list(city_datasets.keys())))

    # Run experiments (use --no-exp1/2/3 to skip individual experiments)
    exp1 = experiment_transfer(city_datasets) if not _ARGS.no_exp1 else {}
    exp2 = experiment_loocv(city_datasets)    if not _ARGS.no_exp2 else {}
    exp3 = experiment_pooled(city_datasets)   if not _ARGS.no_exp3 else ({}, {}, [])

    if not _ARGS.no_exp2:
        plot_loocv(exp2, city_datasets)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("Within-city R2 (from saved RF models):")
    for slug in city_datasets:
        path = os.path.join(MODELS_DIR, "{}_results.csv".format(slug))
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            if "r2_mean" in df.columns:
                row = "  {:<14}".format(slug)
                for mname in ["rf","xgb","gb","ridge"]:
                    if mname in df.index:
                        row += "  {}={:.3f}".format(mname.upper(), df.loc[mname,"r2_mean"])
                print(row)

    if exp2:
        print("\nLOOCV Norm R2 (RF):")
        for slug, v in exp2.items():
            print("  {:<14}: {:.3f}".format(slug, v.get("r2_norm_rf", v.get("r2_norm", 0))))

    if _XGB and exp2:
        print("\nLOOCV Norm R2 (XGB):")
        for slug, v in exp2.items():
            print("  {:<14}: {:.3f}".format(slug, v.get("r2_norm_xgb", 0)))

    rf_mean  = np.mean([v.get("r2_norm_rf",  v.get("r2_norm",0)) for v in exp2.values()]) if exp2 else 0
    xgb_mean = np.mean([v.get("r2_norm_xgb", 0) for v in exp2.values()]) if (_XGB and exp2) else None

    print("\nMean LOOCV Norm R2:")
    print("  RF:  {:.3f}".format(rf_mean))
    if xgb_mean is not None:
        print("  XGB: {:.3f}".format(xgb_mean))
        best = "XGB" if xgb_mean > rf_mean else "RF"
        print("  Best: {}".format(best))

    best_mean = max(rf_mean, xgb_mean or rf_mean)
    if best_mean > 0.3:
        print("-> Unified model VIABLE for MWI pipeline (Norm R2 > 0.3)")
    elif best_mean > 0.1:
        print("-> Partial transfer — city fine-tuning recommended")
    else:
        print("-> City-specific calibration required (one OpeNoise tile per city)")


if __name__ == "__main__":
    main()