"""
Noise Proxy Model Pipeline v5

USAGE:
  # Single city run (reads city from NOISE_FILE name):
  python noise_proxy_pipeline_v5.py

  # Cross-city transfer test:
  python noise_proxy_pipeline_v5.py --transfer noise_contours_pittsburgh.gpkg noise_contours_amsterdam.gpkg

CITY SWITCHING:
  Change only one line: NOISE_FILE = "noise_contours_<city>.gpkg"
  Everything else (city name, CRS, cache, output files) is auto-derived.
"""

import os
import re
import sys
import warnings
import joblib

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    print("  Note: xgboost not installed. Run: pip install xgboost")
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — change only this line to switch cities
# =============================================================================
# =============================================================================
# PIPELINE CONFIGURATION — change NOISE_FILE to switch cities
# =============================================================================
cityname = "bangkok"  # <-- change this to switch cities (e.g. "bangkok", "newyork", "pittsburgh")
NOISE_FILE    = f"noise_contours_{cityname}.gpkg"
NOISE_DB_COL  = "Lden"
BUFFER_DIST_M = 25
OUTPUT_DIR    = "."
MODELS_DIR    = "models"

# =============================================================================
# SHARED FEATURE EXTRACTION — all feature logic lives in noisy_feature_extraction_v1.py
# =============================================================================
from noisy_feature_extraction_v1 import (
    FEATURE_COLS, CITY_NAMES, CITY_CENTRES, HIGHWAY_ORDER,
    DOWNLOAD_RADIUS_M, _FEATURE_COLS_RUNTIME,
    get_osm_data, extract_features, load_and_join_noise,
    get_utm_crs, get_centre_from_noise_file,
)

import re as _re, os as _os
_basename  = _os.path.splitext(_os.path.basename(NOISE_FILE))[0]
_city_slug = _re.sub(r"^noise_contours_", "", _basename)
if _city_slug in CITY_NAMES:
    CITY = CITY_NAMES[_city_slug]
else:
    CITY = _city_slug.replace("_", " ").title()
    CITY_NAMES[_city_slug] = CITY
    print("  Note: auto-resolved '{}' -> '{}'".format(_city_slug, CITY))

CACHE_FILE    = _os.path.join(OUTPUT_DIR, "features_cache_{}.gpkg".format(_city_slug))
RESULTS_FILE  = _os.path.join(OUTPUT_DIR, "segments_noise_{}.gpkg".format(_city_slug))
PLOT_FILE     = _os.path.join(OUTPUT_DIR, "noise_model_results_{}.png".format(_city_slug))


# =============================================================================
# STEP 5: Train, compare, and save models
# =============================================================================
def train_and_evaluate(edges_m, feature_cols, city_slug=None):
    """
    Train RF, GB, Ridge, SVR. Compare via 5-fold CV. Save all models.

    To reload a saved model later:
        bundle = joblib.load("models/amsterdam_rf.joblib")
        model  = bundle["model"]
        feats  = bundle["feature_cols"]
        y_pred = model.predict(X_new)
    """
    if city_slug is None:
        city_slug = _city_slug
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[5/5] Training and comparing models...")
    df = edges_m[feature_cols + ["noise_db"]].dropna()
    X  = df[feature_cols].values
    y  = df["noise_db"].values
    print("      n = {} segments".format(len(df)))
    print("      X shape: {}  |  Y: {:.1f} - {:.1f} dB".format(
        X.shape, y.min(), y.max()))

    candidates = {
        "rf": RandomForestRegressor(
            n_estimators=300, max_features="sqrt",
            min_samples_leaf=5, random_state=42, n_jobs=-1),
        "gb": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0))]),
        "svr": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVR(kernel="rbf", C=10, epsilon=0.5))]),
    }
    if _XGBOOST_AVAILABLE:
        candidates["xgb"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, n_jobs=-1,
            verbosity=0)

    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n  +-- Results (5-fold CV) -------------------------------------------")
    for name, model in candidates.items():
        r2  = cross_val_score(model, X, y, cv=kf, scoring="r2")
        mae = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
        results[name] = {
            "r2_mean": r2.mean(), "r2_std": r2.std(),
            "mae_mean": -mae.mean(), "mae_std": mae.std()
        }
        print("  |  {:<6}  R2: {:.3f} +/- {:.3f}   MAE: {:.2f} +/- {:.2f} dB".format(
            name.upper(), r2.mean(), r2.std(), -mae.mean(), mae.std()))
    print("  +------------------------------------------------------------------\n")

    print("  Saving models...")
    for name, model in candidates.items():
        model.fit(X, y)
        path = os.path.join(MODELS_DIR, "{}_{}.joblib".format(city_slug, name))
        joblib.dump({
            "model":        model,
            "feature_cols": feature_cols,
            "city":         CITY,
            "city_slug":    city_slug,
            "cv_results":   results[name],
        }, path)
        print("    Saved -> {}".format(path))

    meta_path = os.path.join(MODELS_DIR, "{}_results.csv".format(city_slug))
    pd.DataFrame(results).T.to_csv(meta_path)
    print("    Results summary -> {}".format(meta_path))

    rf = candidates["rf"]
    importances = pd.Series(
        rf.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print("\n  Feature importances (RF):")
    for feat, imp in importances.items():
        bar = "#" * int(imp * 40)
        print("    {:<25} {:.3f}  {}".format(feat, imp, bar))

    y_pred    = cross_val_predict(rf, X, y, cv=5)
    residuals = y - y_pred
    return rf, importances, df, y_pred, residuals


# =============================================================================
# STEP 6: Plots and export
# =============================================================================
def plot_results(importances, df, feature_cols, y_pred, residuals):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Noise Proxy Model ({}) -- OSM Morphology -> Modeled dB (Lden)".format(_city_slug),
        fontsize=13, y=1.02
    )

    importances.sort_values().plot(kind="barh", ax=axes[0], color="#028090")
    axes[0].set_title("Feature Importance (RF)")
    axes[0].set_xlabel("Mean Decrease Impurity")

    lim = [df["noise_db"].min() - 2, df["noise_db"].max() + 2]
    axes[1].scatter(df["noise_db"], y_pred, alpha=0.25, s=8, color="#028090")
    axes[1].plot(lim, lim, "r--", linewidth=1, label="1:1 line")
    axes[1].set_xlabel("OpeNoise modeled dB (Lden)")
    axes[1].set_ylabel("RF predicted dB")
    axes[1].set_title("Predicted vs OpeNoise (5-fold CV)")
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].legend()

    axes[2].hist(residuals, bins=40, color="#028090", edgecolor="white", linewidth=0.3)
    axes[2].axvline(0, color="red", linewidth=1, linestyle="--")
    axes[2].set_xlabel("Residual (actual - predicted) dB")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Residuals  (mean={:.2f}, std={:.2f} dB)".format(
        residuals.mean(), residuals.std()))

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print("  Plot saved -> {}".format(PLOT_FILE))


def export_gpkg(edges_m, feature_cols, rf):
    out  = edges_m[["geometry", "noise_db"] + feature_cols].copy()
    out  = out.to_crs("EPSG:4326")
    mask = out[feature_cols].notna().all(axis=1)
    out["noise_predicted"] = np.nan
    out.loc[mask, "noise_predicted"] = rf.predict(out.loc[mask, feature_cols].values)
    out.to_file(RESULTS_FILE, driver="GPKG")
    print("  GeoPackage saved -> {}".format(RESULTS_FILE))


# =============================================================================
# MAIN RUN
# =============================================================================
def plot_within_city_comparison(edges_m, feature_cols, rf):
    """
    Two separate square PNG files for use as individual journal subfigures:
      within_city_comparison_<city>_pred.png  — (a) model prediction
      within_city_comparison_<city>_gt.png    — (b) OpeNoise ground truth

    Design choices (journal-ready):
      - White background throughout
      - Blue-to-red colourmap (RdYlBu_r): blue=quiet, red=loud
      - Thin black street outlines for legibility on white
      - Per-panel horizontal colourbar at bottom, fixed 0-99 dB scale
      - No title on the figure (use LaTeX caption instead)
      - Stats annotation (R2/MAE or mean/std) in bottom-left corner
    """
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    from sklearn.metrics import r2_score, mean_absolute_error

    print("  Generating within-city comparison maps (2 separate files)...")

    matched = edges_m.copy()
    feat_mask = (matched[feature_cols].notna().all(axis=1)
                 & matched["noise_db"].notna())
    matched   = matched[feat_mask].copy()
    matched   = matched.to_crs("EPSG:4326")
    matched["noise_predicted"] = rf.predict(matched[feature_cols].values)

    r2_all  = r2_score(matched["noise_db"], matched["noise_predicted"])
    mae_all = mean_absolute_error(matched["noise_db"], matched["noise_predicted"])
    bias    = (matched["noise_predicted"] - matched["noise_db"]).mean()

    # Fixed 0-99 dB scale — consistent with validate_prediction.py
    # and detroit_validation.py across the whole project
    vmin, vmax = 0, 99

    # Blue-to-red: cool=quiet, warm=loud
    cmap = plt.cm.RdYlBu_r

    # Linewidths by highway class — thin for print clarity
    hw_lw = {9:1.8, 8:1.5, 7:1.3, 6:1.1, 5:0.9,
              4:0.7, 3:0.6, 2:0.45, 1:0.35, 0:0.25}
    outline_extra = 0.4   # extra width for black outline pass

    panels = [
        ("noise_predicted", "pred",
         "R\u00b2={:.3f}  MAE={:.2f} dB  bias={:+.2f} dB".format(
             r2_all, mae_all, bias)),
        ("noise_db", "gt",
         "mean={:.1f} dB  std={:.1f} dB".format(
             matched["noise_db"].mean(), matched["noise_db"].std())),
    ]

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for col, suffix, stat_text in panels:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_axis_off()

        if "highway_class" in matched.columns:
            for hw in sorted(matched["highway_class"].dropna().unique()):
                sub = matched[matched["highway_class"] == hw].copy()
                lw  = hw_lw.get(int(hw), 0.3)
                # Pass 1: thin black outline
                sub.plot(ax=ax, color="black",
                         linewidth=lw + outline_extra,
                         alpha=0.18, zorder=int(hw))
                # Pass 2: colour fill
                clrs = [cmap(norm(v)) for v in sub[col].clip(vmin, vmax)]
                sub.plot(ax=ax, color=clrs, linewidth=lw,
                         alpha=0.95, zorder=int(hw) + 1)
        else:
            clrs = [cmap(norm(v)) for v in matched[col].clip(vmin, vmax)]
            matched.plot(ax=ax, color="black", linewidth=0.6,
                         alpha=0.15, zorder=1)
            matched.plot(ax=ax, color=clrs, linewidth=0.5,
                         alpha=0.95, zorder=2)

        # Tighten map extent to actual data bounds — removes blank padding
        bounds = matched.total_bounds  # [minx, miny, maxx, maxy]
        pad_x  = (bounds[2] - bounds[0]) * 0.01
        pad_y  = (bounds[3] - bounds[1]) * 0.01
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

        # Source label between map and colourbar — one line, no box
        source_text = {
            "pred": "Source: OSM morphology model (this study)",
            "gt":   "Source: OpeNoise CNOSSOS-EU simulation",
        }[suffix]

        # Per-panel horizontal colourbar outside map, fixed 0-99 dB
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.subplots_adjust(bottom=0.16)
        cbar_ax = fig.add_axes([0.08, 0.06, 0.84, 0.025])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("L$_{{den}}$ (dBA)", fontsize=8.5, labelpad=4,
                       color="#222222")
        cbar.ax.xaxis.set_tick_params(color="#222222", labelsize=7.5)
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color="#222222")
        cbar.set_ticks(range(0, 100, 10))
        cbar.set_ticklabels([str(t) for t in range(0, 100, 10)])

        # Source label sits just above the colourbar
        fig.text(0.5, 0.145, source_text,
                 ha="center", va="bottom", fontsize=8,
                 color="#555555", style="italic")

        # No title — caption handles this in the paper
        plt.tight_layout(rect=[0, 0.13, 1, 1])
        out = os.path.join(OUTPUT_DIR,
                           "within_city_comparison_{}_{}.png".format(
                               _city_slug, suffix))
        if os.path.exists(out):
            os.remove(out)
        plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print("  Saved -> {}".format(out))


def run(use_cache=True):
    print("Running Noise Proxy Pipeline v5 for: {}".format(CITY))

    if use_cache and os.path.exists(CACHE_FILE):
        print("Loading features from cache: {}".format(CACHE_FILE))
        edges_m      = gpd.read_file(CACHE_FILE)
        crs_metric   = edges_m.crs.to_string()
        feature_cols = _FEATURE_COLS_RUNTIME
        print("      {} segments loaded from cache.".format(len(edges_m)))
    else:
        if use_cache:
            print("Cache not found at {} -- running full pipeline.".format(CACHE_FILE))
        crs_metric, lat, lon      = get_utm_crs(noise_file=NOISE_FILE)
        edges, buildings, green, water, G = get_osm_data(CITY, lat, lon)
        edges_m, feature_cols = extract_features(edges, buildings, green, water, G, crs_metric)
        edges_m.to_file(CACHE_FILE, driver="GPKG")
        import stat
        os.chmod(CACHE_FILE, stat.S_IRUSR | stat.S_IWUSR |
                              stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH)
        print("      Features cached -> {}".format(CACHE_FILE))

    edges_m = load_and_join_noise(
        edges_m, NOISE_FILE, NOISE_DB_COL, BUFFER_DIST_M, crs_metric
    )
    rf, importances, df, y_pred, resid = train_and_evaluate(edges_m, feature_cols)
    plot_results(importances, df, feature_cols, y_pred, resid)
    plot_within_city_comparison(edges_m, feature_cols, rf)
    export_gpkg(edges_m, feature_cols, rf)
    print("\nDone.")
    return rf, edges_m


# =============================================================================
# CROSS-CITY TRANSFER
# =============================================================================
def cross_city_transfer(source_noise_file, target_noise_file, use_cache=True):
    """
    Train on all segments from source city, test on all from target city.
    No data leakage — zero overlap between train and test.

    Usage:
        python noise_proxy_pipeline_v5.py --transfer \
            noise_contours_pittsburgh.gpkg noise_contours_amsterdam.gpkg
    """
    print("=" * 62)
    print("CROSS-CITY TRANSFER TEST")

    def load_city(noise_file):
        slug  = re.sub(r"^noise_contours_", "",
                       os.path.splitext(os.path.basename(noise_file))[0])
        city  = CITY_NAMES.get(slug, slug.replace("_", ", ").title())
        cache = os.path.join(OUTPUT_DIR, "features_cache_{}.gpkg".format(slug))

        print("\n  Loading {} ({})...".format(city, slug))

        if use_cache and os.path.exists(cache):
            em  = gpd.read_file(cache)
            crs = em.crs.to_string()
            print("    {} segments from cache".format(len(em)))
        else:
            crs, lat, lon          = get_utm_crs(noise_file=noise_file)
            edges, bld, grn, wat, G_c = get_osm_data(city, lat, lon)
            em, _                  = extract_features(edges, bld, grn, wat, G_c, crs)
            em.to_file(cache, driver="GPKG")
            # Ensure file is writable for future reads (pyogrio needs write access)
            import stat
            os.chmod(cache, stat.S_IRUSR | stat.S_IWUSR |
                            stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH)
            print("    {} segments cached -> {}".format(len(em), cache))

        em = load_and_join_noise(em, noise_file, NOISE_DB_COL, BUFFER_DIST_M, crs)
        df = em[_FEATURE_COLS_RUNTIME + ["noise_db"]].dropna()
        print("    {} segments with noise labels".format(len(df)))
        return df, slug

    df_src, src_slug = load_city(source_noise_file)
    df_tgt, tgt_slug = load_city(target_noise_file)

    X_src, y_src = df_src[_FEATURE_COLS_RUNTIME].values, df_src["noise_db"].values
    X_tgt, y_tgt = df_tgt[_FEATURE_COLS_RUNTIME].values, df_tgt["noise_db"].values

    rf = RandomForestRegressor(
        n_estimators=300, max_features="sqrt",
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )

    print("\n  Train -> {} ({} segments)".format(src_slug, len(X_src)))
    print("  Test  -> {} ({} segments)".format(tgt_slug, len(X_tgt)))

    rf.fit(X_src, y_src)
    r2_fwd  = r2_score(y_tgt, rf.predict(X_tgt))
    mae_fwd = mean_absolute_error(y_tgt, rf.predict(X_tgt))

    rf.fit(X_tgt, y_tgt)
    r2_rev  = r2_score(y_src, rf.predict(X_src))
    mae_rev = mean_absolute_error(y_src, rf.predict(X_src))

    from sklearn.preprocessing import QuantileTransformer

    def _tr(Xs, ys, Xt, yt):
        rf.fit(Xs, ys)
        return r2_score(yt, rf.predict(Xt)), mean_absolute_error(yt, rf.predict(Xt))

    # ── Test 1: Raw ───────────────────────────────────────────────────────────
    r2_fwd_raw,  mae_fwd_raw  = _tr(X_src, y_src, X_tgt, y_tgt)
    r2_rev_raw,  mae_rev_raw  = _tr(X_tgt, y_tgt, X_src, y_src)

    # ── Test 2: Z-score normalised ────────────────────────────────────────────
    # Removes city-level mean/variance from features AND target.
    # Tests whether relative noise patterns transfer once city offsets removed.
    def _zn(arr2d):
        mu = arr2d.mean(axis=0); sd = arr2d.std(axis=0) + 1e-9
        return (arr2d - mu) / sd
    def _zv(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    Xs_z, ys_z = _zn(X_src), _zv(y_src)
    Xt_z, yt_z = _zn(X_tgt), _zv(y_tgt)
    r2_fwd_z, mae_fwd_z = _tr(Xs_z, ys_z, Xt_z, yt_z)
    r2_rev_z, mae_rev_z = _tr(Xt_z, yt_z, Xs_z, ys_z)

    # ── Test 3: Rank (quantile) ───────────────────────────────────────────────
    qt_s = QuantileTransformer(output_distribution="normal", random_state=42).fit(X_src)
    qt_t = QuantileTransformer(output_distribution="normal", random_state=42).fit(X_tgt)
    Xs_q, Xt_q = qt_s.transform(X_src), qt_t.transform(X_tgt)
    r2_fwd_q, mae_fwd_q = _tr(Xs_q, ys_z, Xt_q, yt_z)
    r2_rev_q, mae_rev_q = _tr(Xt_q, yt_z, Xs_q, ys_z)

    # ── Print ─────────────────────────────────────────────────────────────────
    fw = "{}->{}".format(src_slug, tgt_slug)
    rv = "{}->{}".format(tgt_slug, src_slug)
    print("\n  +-- Cross-City Transfer Results " + "-"*46)
    print("  |  {:<20}  {:>22}  {:>22}".format("Method", fw, rv))
    print("  |  " + "-"*68)
    print("  |  {:<20}  R2={:>6.3f} MAE={:>5.1f}dB  R2={:>6.3f} MAE={:>5.1f}dB".format(
        "Raw (absolute)", r2_fwd_raw, mae_fwd_raw, r2_rev_raw, mae_rev_raw))
    print("  |  {:<20}  R2={:>6.3f} MAE={:>5.2f}sd  R2={:>6.3f} MAE={:>5.2f}sd".format(
        "Z-score norm.", r2_fwd_z, mae_fwd_z, r2_rev_z, mae_rev_z))
    print("  |  {:<20}  R2={:>6.3f} MAE={:>5.2f}sd  R2={:>6.3f} MAE={:>5.2f}sd".format(
        "Rank (quantile)", r2_fwd_q, mae_fwd_q, r2_rev_q, mae_rev_q))
    print("  +" + "-"*78)
    print()
    print("  Interpretation:")
    print("  Raw R2<0     -> absolute dB scale is city-specific (expected)")
    print("  Z-norm R2>0  -> relative noise gradients transfer after city offset removed")
    print("  Rank R2>0    -> noise ranking by morphology transfers regardless of scale")
    print()

    # Feature importances on source model
    rf.fit(X_src, y_src)
    imps = pd.Series(
        rf.feature_importances_, index=_FEATURE_COLS_RUNTIME
    ).sort_values(ascending=False)
    print("  Feature importances (trained on {}):".format(src_slug))
    for feat, imp in imps.items():
        bar = "#" * int(imp * 40)
        print("    {:<25} {:.3f}  {}".format(feat, imp, bar))

    # Scatter plots
    r2_fwd, mae_fwd = r2_fwd_raw, mae_fwd_raw
    r2_rev, mae_rev = r2_rev_raw, mae_rev_raw

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cross-City Transfer: {} vs {} (raw dB)".format(
        src_slug, tgt_slug), fontsize=13)

    rf.fit(X_src, y_src)
    p1   = rf.predict(X_tgt)
    lim1 = [y_tgt.min() - 2, y_tgt.max() + 2]
    axes[0].scatter(y_tgt, p1, alpha=0.2, s=8, color="#028090")
    axes[0].plot(lim1, lim1, "r--", lw=1)
    axes[0].set_xlabel("OpeNoise dB ({})".format(tgt_slug))
    axes[0].set_ylabel("Predicted dB")
    axes[0].set_title("Train:{} Test:{}  Raw={:.3f}  Norm={:.3f}".format(
        src_slug, tgt_slug, r2_fwd_raw, r2_fwd_z))

    rf.fit(X_tgt, y_tgt)
    p2   = rf.predict(X_src)
    lim2 = [y_src.min() - 2, y_src.max() + 2]
    axes[1].scatter(y_src, p2, alpha=0.2, s=8, color="#c05000")
    axes[1].plot(lim2, lim2, "r--", lw=1)
    axes[1].set_xlabel("OpeNoise dB ({})".format(src_slug))
    axes[1].set_ylabel("Predicted dB")
    axes[1].set_title("Train:{} Test:{}  Raw={:.3f}  Norm={:.3f}".format(
        tgt_slug, src_slug, r2_rev_raw, r2_rev_z))

    plt.tight_layout()
    plot_path = os.path.join(
        OUTPUT_DIR, "transfer_{}_vs_{}.png".format(src_slug, tgt_slug))
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print("  Plot saved -> {}".format(plot_path))
    print("Cross-city transfer complete.")

    return {
        "{}_to_{}".format(src_slug, tgt_slug): {
            "r2_raw": r2_fwd_raw, "r2_norm": r2_fwd_z,
            "r2_rank": r2_fwd_q,  "mae_raw": mae_fwd_raw},
        "{}_to_{}".format(tgt_slug, src_slug): {
            "r2_raw": r2_rev_raw, "r2_norm": r2_rev_z,
            "r2_rank": r2_rev_q,  "mae_raw": mae_rev_raw},
    }



# =============================================================================
# MULTI-CITY UNIFIED MODEL
# =============================================================================
def multi_city_train(noise_files, use_cache=True):
    """
    Train a unified model on all cities combined.
    Evaluates with leave-one-city-out cross-validation.

    Two training strategies:
      1. Raw pooling       -- concatenate all cities as-is
      2. Z-normalised      -- normalise dB within each city before pooling

    Usage:
        python noise_proxy_pipeline_v5.py --multi \
            noise_contours_pittsburgh.gpkg \
            noise_contours_amsterdam.gpkg \
            noise_contours_bangkok.gpkg \
            noise_contours_newyork.gpkg
    """
    print("=" * 62)
    print("MULTI-CITY UNIFIED MODEL -- Leave-One-City-Out CV")
    print("=" * 62)

    # Load all cities
    city_data = {}
    for nf in noise_files:
        slug  = re.sub(r"^noise_contours_", "",
                       os.path.splitext(os.path.basename(nf))[0])
        city  = CITY_NAMES.get(slug, slug)
        cache = os.path.join(OUTPUT_DIR, "features_cache_{}.gpkg".format(slug))

        print("\n  Loading {}...".format(slug))
        if use_cache and os.path.exists(cache):
            em  = gpd.read_file(cache)
            crs = em.crs.to_string()
            print("    {} segments from cache".format(len(em)))
        else:
            crs, lat, lon        = get_utm_crs(noise_file=nf)
            edges, bld, grn, Gc  = get_osm_data(city, lat, lon)
            em, _                = extract_features(edges, bld, grn, Gc, crs)
            em.to_file(cache, driver="GPKG")

        em = load_and_join_noise(em, nf, NOISE_DB_COL, BUFFER_DIST_M, crs)
        df = em[_FEATURE_COLS_RUNTIME + ["noise_db"]].dropna()
        print("    {} labelled segments".format(len(df)))
        city_data[slug] = df

    if len(city_data) < 2:
        print("Need at least 2 cities.")
        return

    rf    = RandomForestRegressor(
        n_estimators=300, max_features="sqrt",
        min_samples_leaf=5, random_state=42, n_jobs=-1)
    slugs = list(city_data.keys())

    def _zv(v):
        return (v - v.mean()) / (v.std() + 1e-9)
    def _zn(arr):
        mu = arr.mean(axis=0); sd = arr.std(axis=0) + 1e-9
        return (arr - mu) / sd

    results = {}
    print("\n  +-- Leave-One-City-Out Results " + "-"*32)
    print("  |  {:<12}  {:>8}  {:>8}  {:>8}  {:>7}".format(
        "Test city", "Raw R2", "Norm R2", "Raw MAE", "n_test"))
    print("  |  " + "-"*52)

    for test_slug in slugs:
        train_dfs = [city_data[s] for s in slugs if s != test_slug]
        test_df   = city_data[test_slug]
        df_train  = pd.concat(train_dfs, ignore_index=True)

        X_tr = df_train[_FEATURE_COLS_RUNTIME].values
        y_tr = df_train["noise_db"].values
        X_te = test_df[_FEATURE_COLS_RUNTIME].values
        y_te = test_df["noise_db"].values

        # Raw
        rf.fit(X_tr, y_tr)
        r2_raw  = r2_score(y_te, rf.predict(X_te))
        mae_raw = mean_absolute_error(y_te, rf.predict(X_te))

        # Z-normalised pooling
        X_tr_z = np.vstack([_zn(city_data[s][_FEATURE_COLS_RUNTIME].values)
                            for s in slugs if s != test_slug])
        y_tr_z = np.concatenate([_zv(city_data[s]["noise_db"].values)
                                 for s in slugs if s != test_slug])
        X_te_z = _zn(X_te)
        y_te_z = _zv(y_te)

        rf.fit(X_tr_z, y_tr_z)
        r2_norm = r2_score(y_te_z, rf.predict(X_te_z))

        results[test_slug] = {"r2_raw": r2_raw, "r2_norm": r2_norm,
                               "mae_raw": mae_raw, "n_test": len(y_te)}
        print("  |  {:<12}  {:>8.3f}  {:>8.3f}  {:>8.2f}  {:>7}".format(
            test_slug, r2_raw, r2_norm, mae_raw, len(y_te)))

    r2_raw_mean  = np.mean([v["r2_raw"]  for v in results.values()])
    r2_norm_mean = np.mean([v["r2_norm"] for v in results.values()])
    mae_mean     = np.mean([v["mae_raw"] for v in results.values()])
    print("  |  " + "-"*52)
    print("  |  {:<12}  {:>8.3f}  {:>8.3f}  {:>8.2f}".format(
        "MEAN", r2_raw_mean, r2_norm_mean, mae_mean))
    print("  +" + "-"*54)
    print()
    print("  Interpretation:")
    print("  Raw R2>0    -> unified model generalises without normalisation")
    print("  Norm R2>0   -> relative noise patterns transfer across cities")
    print("  Norm R2>0.3 -> unified normalised model viable for MWI pipeline")

    out = os.path.join(OUTPUT_DIR, "multi_city_results.csv")
    pd.DataFrame(results).T.to_csv(out)
    print("  Results saved -> {}".format(out))
    return results

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if "--transfer" in sys.argv:
        idx = sys.argv.index("--transfer")
        src = sys.argv[idx + 1]
        tgt = sys.argv[idx + 2]
        cross_city_transfer(src, tgt, use_cache=True)

    elif "--multi" in sys.argv:
        idx   = sys.argv.index("--multi")
        files = sys.argv[idx + 1:]
        multi_city_train(files, use_cache=True)

    elif "--plot-only" in sys.argv:
        # ── Fast replot: load cache + saved RF, skip all extraction/training ──
        # Usage:
        #   python noise_proxy_pipeline_v5.py --plot-only singapore
        #   python noise_proxy_pipeline_v5.py --plot-only           (uses ACTIVE_CITY)
        import argparse as _ap
        _p = _ap.ArgumentParser()
        _p.add_argument("--plot-only", action="store_true")
        _p.add_argument("city", nargs="?", default=None)
        _args = _p.parse_args()

        # Override city globals if slug provided
        if _args.city:
            _city_slug_arg = _args.city.lower()
            if _city_slug_arg in CITY_NAMES:
                globals()["_city_slug"] = _city_slug_arg
                globals()["CITY"]       = CITY_NAMES[_city_slug_arg]
            else:
                globals()["_city_slug"] = _city_slug_arg
                globals()["CITY"]       = _city_slug_arg.replace("_", " ").title()
            # Update file paths that depend on city slug
            globals()["CACHE_FILE"]   = os.path.join(
                OUTPUT_DIR, "features_cache_{}.gpkg".format(_city_slug))
            globals()["RESULTS_FILE"] = os.path.join(
                OUTPUT_DIR, "segments_noise_{}.gpkg".format(_city_slug))

        print("=" * 62)
        print("PLOT-ONLY MODE: {}".format(CITY))
        print("Skipping feature extraction and training.")
        print("=" * 62)

        # Load cached features
        if not os.path.exists(CACHE_FILE):
            print("ERROR: feature cache not found: {}".format(CACHE_FILE))
            print("  Run the full pipeline first:")
            print("  python noise_proxy_pipeline_v5.py --city {}".format(_city_slug))
            sys.exit(1)
        print("[1/3] Loading feature cache: {}".format(CACHE_FILE))
        edges_m      = gpd.read_file(CACHE_FILE)
        feature_cols = _FEATURE_COLS_RUNTIME
        print("      {} segments".format(len(edges_m)))

        # Derive noise file from city slug — same convention as full pipeline
        city_noise_file = os.path.join(
            OUTPUT_DIR, "noise_contours_{}.gpkg".format(_city_slug))
        if not os.path.exists(city_noise_file):
            # Fallback: check current directory
            city_noise_file = "noise_contours_{}.gpkg".format(_city_slug)
        if not os.path.exists(city_noise_file):
            print("ERROR: noise contours not found: {}".format(city_noise_file))
            print("  Expected: noise_contours_{}.gpkg".format(_city_slug))
            sys.exit(1)
        print("[2/3] Loading noise labels: {}".format(city_noise_file))
        crs_metric = edges_m.crs.to_string()
        edges_m = load_and_join_noise(
            edges_m, city_noise_file, NOISE_DB_COL, BUFFER_DIST_M, crs_metric)
        n_matched = edges_m["noise_db"].notna().sum()
        print("      {}/{} segments matched".format(n_matched, len(edges_m)))
        if n_matched == 0:
            print("ERROR: 0 segments matched — noise tile and feature cache")
            print("  may have been generated from different city centres.")
            print("  Re-run the full pipeline: python noise_proxy_pipeline_v5.py --city {}".format(_city_slug))
            sys.exit(1)

        # Load saved RF model
        rf_path = os.path.join(MODELS_DIR, "{}_rf.joblib".format(_city_slug))
        if not os.path.exists(rf_path):
            print("ERROR: RF model not found: {}".format(rf_path))
            print("  Run the full pipeline first to train and save models.")
            sys.exit(1)
        print("[3/3] Loading RF model: {}".format(rf_path))
        _bundle = joblib.load(rf_path)
        rf      = _bundle["model"]
        # Use feature cols from bundle if available (safer)
        if "feature_cols" in _bundle:
            feature_cols = _bundle["feature_cols"]

        # Replot
        plot_within_city_comparison(edges_m, feature_cols, rf)
        print("\nDone. Output files:")
        print("  within_city_comparison_{}_pred.png".format(_city_slug))
        print("  within_city_comparison_{}_gt.png".format(_city_slug))

    else:
        run(use_cache=True)