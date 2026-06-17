"""Do GEDI and VSM carry information about LVIS-defined lower/sub-canopy
structure, *after* canopy-top height is controlled?

The question is deliberately NOT "can VSM reconstruct the full LVIS profile".
It is a falsifiable classification test:

> Within matched canopy-top-height bins, can GEDI / VSM tell apart footprints
> that LVIS says have a DENSE vs a SPARSE lower canopy?

Because the dense/sparse label is defined *inside* LVIS RH98 height bins, the
label is (by construction) independent of canopy-top height — so any classifier
that beats its own height-only baseline is using sub-canopy structure, not
height.

Pipeline (run once per sub-canopy definition; default RH25/RH98 and RH50/RH98):

  1. Pool the per-tile LVIS-GEDI-VSM pair parquets. Keep footprints with valid
     (finite, > rh98_min) RH98 on ALL three sensors. Tile = filename stem (S2
     MGRS) — the spatial group for cross-validation.
  2. Bin footprints by LVIS RH98 (10-20, 20-30, 30-40, 40-50, >50 m).
  3. Within each height bin: LVIS_sub = lvis_RH<num> / lvis_RH98. Label the top
     `q` fraction "dense" (1), the bottom `q` "sparse" (0), discard the middle.
     Record the binary label and the source height bin.
  4. For each sensor (GEDI, VSM) and each feature set
       height_only  = [RH98]
       shape_only   = [RH25/RH98, RH50/RH98, RH75/RH98]
       height_shape = [RH98, RH25/RH98, RH50/RH98, RH75/RH98]
       extended     = height_shape + [FHD, ENL, CR]   (only if those cols exist)
     run the SAME classifier (logreg or random_forest) under spatially
     independent CV (GroupKFold by tile, or leave-one-tile-out). Never let a
     tile straddle train/test.
  5. Report ROC-AUC, balanced accuracy and F1 as mean +/- SD across folds, plus
     pooled out-of-fold ROC curves, feature importances, confusion matrices and
     per-height-bin AUC.
  6. Headline comparison per sensor:
       Delta_AUC = AUC(height_shape) - AUC(height_only)
       Delta_BA  = BA(height_shape)  - BA(height_only)
     and GEDI-vs-VSM signal strength.

Data: the per-tile LVIS-GEDI-VSM pair parquets from
`evaluation/on_lvis.extract_vsm_on_pair_locations`
(`Gabon2016_vs_GEDI2020_stable_forest_with_vsm2020/<tile>.parquet`), columns
`lvis_RH<NN>` / `vsm_RH<NN>` (uppercase) and `gedi_rh<NN>` (lowercase). FHD/ENL/
CR are optional and skipped if absent.

  python -m tools.run run=subcanopy_classify
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from const import FIGURE_SIZES, FONT_SIZES

# Canonical RH levels used for the shape features and the canopy-top normalizer.
_SHAPE_LEVELS = (25, 50, 75)
_TOP = 98
# Sub-canopy numerators to define the dense/sparse label (one full run each).
_DEFAULT_DEFS = (25, 50)


def _rh_col(sensor: str, level: int) -> str:
    """Pair-parquet RH column for one sensor/level. LVIS & VSM are uppercase
    `RH<NN>`; GEDI is lowercase dense `rh<NN>`."""
    if sensor == 'gedi':
        return f'gedi_rh{level}'
    return f'{sensor}_RH{level}'


# Extra (optional) structural columns for the extended model, per sensor.
def _extra_cols(sensor: str) -> dict:
    return {'FHD': f'{sensor}_fhd', 'ENL': f'{sensor}_enl', 'CR': f'{sensor}_cr'}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_pairs(pairs_dir: Path, glob: str, sensors: tuple, def_levels: tuple,
                say, extra_cols: tuple = ()) -> pd.DataFrame:
    """Pool the per-tile pair parquets into one frame holding every RH column we
    need (LVIS label levels + each sensor's RH25/50/75/98) plus any available
    extended columns. Stamps each row with `tile` (filename stem). Fails loudly
    on a missing *required* RH column; extended columns are detected, not
    required.

    `extra_cols` are additional literal column names (e.g. `lvis_COMPLEXITY`)
    that are required and passed through unchanged — use for non-RH metrics a
    caller needs alongside the RH columns."""
    files = sorted(pairs_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f'No files matching {glob!r} under {pairs_dir}')

    lvis_levels = sorted({_TOP} | set(def_levels))
    required = [_rh_col('lvis', lv) for lv in lvis_levels]
    for s in sensors:
        required += [_rh_col(s, lv) for lv in (*_SHAPE_LEVELS, *def_levels, _TOP)]
    required = sorted(set(required) | set(extra_cols))

    available = set(pq.ParquetFile(files[0]).schema.names)
    missing = sorted(set(required) - available)
    if missing:
        raise KeyError(
            f'Pair parquet {files[0].name} is missing required columns '
            f'{missing}.\nAvailable columns:\n  {sorted(available)}\n'
            f'Align column names before rerunning (do not guess).')

    # Optional extended columns: keep only those present on ALL sensors.
    extra_present = {}
    for s in sensors:
        cols = _extra_cols(s)
        have = {k: c for k, c in cols.items() if c in available}
        extra_present[s] = have
    extra_ok = {k for k in ('FHD', 'ENL', 'CR')
                if all(k in extra_present[s] for s in sensors)}
    read_cols = list(required)
    for s in sensors:
        read_cols += [extra_present[s][k] for k in extra_ok]
    read_cols = sorted(set(read_cols))

    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=read_cols)
        if d.empty:
            continue
        d = d.copy()
        d['tile'] = f.stem
        parts.append(d)
    if not parts:
        raise ValueError(f'All {len(files)} pair parquets under {pairs_dir} '
                         f'were empty.')
    df = pd.concat(parts, ignore_index=True)
    say(f'Loaded {len(df):,} footprints from {len(files)} tile(s); '
        f'{df["tile"].nunique()} unique tiles.')
    say(f'Extended structural columns present on all sensors: '
        f'{sorted(extra_ok) if extra_ok else "none (extended model skipped)"}.')
    return df, sorted(extra_ok)


def _clean(df: pd.DataFrame, sensors: tuple, def_levels: tuple, rh98_min: float,
           say) -> pd.DataFrame:
    """Clip negative RH to 0 and keep footprints with finite, > rh98_min RH98 on
    ALL three sensors and finite shape/label RH. Objective: 'only footprints
    with valid RH98 for all sensors are retained'."""
    work = df.replace([np.inf, -np.inf], np.nan).copy()
    rh_cols = [_rh_col('lvis', lv) for lv in sorted({_TOP} | set(def_levels))]
    for s in sensors:
        rh_cols += [_rh_col(s, lv) for lv in (*_SHAPE_LEVELS, *def_levels, _TOP)]
    rh_cols = sorted(set(rh_cols))
    for c in rh_cols:
        work[c] = work[c].clip(lower=0)

    n0 = len(work)
    keep = work[rh_cols].notna().all(axis=1)
    for s in ('lvis', *sensors):
        keep &= work[_rh_col(s, _TOP)] > rh98_min
    out = work[keep].reset_index(drop=True)
    say(f'Cleaning: {n0:,} footprints; dropped {int((~keep).sum()):,} '
        f'(non-finite or RH98 <= {rh98_min:g} m on any sensor) -> '
        f'{len(out):,} kept.')
    return out


# ---------------------------------------------------------------------------
# Labelling: dense vs sparse sub-canopy, defined inside RH98 height bins
# ---------------------------------------------------------------------------
def _make_labels(df: pd.DataFrame, num_level: int, height_edges: tuple,
                 q: float, min_bin_n: int, say) -> pd.DataFrame:
    """Add `lvis_sub`, `height_bin`, `label` (1=dense, 0=sparse, NaN=discarded).
    Terciles are taken WITHIN each LVIS RH98 height bin so the label is
    independent of canopy-top height."""
    out = df.copy()
    top = out[_rh_col('lvis', _TOP)].to_numpy()
    num = out[_rh_col('lvis', num_level)].to_numpy()
    out['lvis_sub'] = num / top

    edges = list(height_edges) + [np.inf]
    out['height_bin'] = ''
    out['label'] = np.nan
    say(f'Sub-canopy definition: LVIS_RH{num_level} / LVIS_RH98; '
        f'dense = top {q:.0%}, sparse = bottom {q:.0%} within each height bin.')
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (top >= lo) & (top < hi)
        label = f'{lo:g}+' if not np.isfinite(hi) else f'{lo:g}-{hi:g}'
        nbin = int(m.sum())
        if nbin < min_bin_n:
            say(f'  [{label:>7}] N={nbin:>6} -> skipped (< min_bin_n={min_bin_n})')
            continue
        sub = out.loc[m, 'lvis_sub']
        qlo, qhi = sub.quantile(q), sub.quantile(1 - q)
        dense = m & (out['lvis_sub'] >= qhi)
        sparse = m & (out['lvis_sub'] <= qlo)
        out.loc[m, 'height_bin'] = label
        out.loc[dense, 'label'] = 1.0
        out.loc[sparse, 'label'] = 0.0
        say(f'  [{label:>7}] N={nbin:>6} | sub q{q:.0%}={qlo:.3f} '
            f'q{1-q:.0%}={qhi:.3f} -> dense={int(dense.sum())} '
            f'sparse={int(sparse.sum())} (mid {int((m.sum()-dense.sum()-sparse.sum()))} discarded)')

    labelled = out[out['label'].notna()].reset_index(drop=True)
    labelled['label'] = labelled['label'].astype(int)
    say(f'  Labelled footprints: {len(labelled):,} '
        f'({int((labelled["label"]==1).sum())} dense / '
        f'{int((labelled["label"]==0).sum())} sparse) across '
        f'{labelled["height_bin"].nunique()} height bins, '
        f'{labelled["tile"].nunique()} tiles.')
    return labelled


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def _build_features(df: pd.DataFrame, sensor: str, extra_ok: list) -> dict:
    """Return {feature_set_name: DataFrame} for one sensor. Ratios are
    RH<lv>/RH98; columns are named so importances are interpretable."""
    top = df[_rh_col(sensor, _TOP)].to_numpy()
    cols = {}
    cols[f'{sensor}_RH98'] = top
    for lv in _SHAPE_LEVELS:
        cols[f'{sensor}_RH{lv}_RH98'] = df[_rh_col(sensor, lv)].to_numpy() / top
    base = pd.DataFrame(cols, index=df.index)

    height = base[[f'{sensor}_RH98']]
    shape = base[[f'{sensor}_RH{lv}_RH98' for lv in _SHAPE_LEVELS]]
    height_shape = base[[f'{sensor}_RH98',
                         *[f'{sensor}_RH{lv}_RH98' for lv in _SHAPE_LEVELS]]]
    sets = {
        'height_only': height,
        'shape_only': shape,
        'height_shape': height_shape,
    }
    if extra_ok:
        ext = height_shape.copy()
        for k in extra_ok:
            c = _extra_cols(sensor)[k]
            ext[f'{sensor}_{k}'] = df[c].to_numpy()
        sets['extended'] = ext
    return sets


# ---------------------------------------------------------------------------
# Classifier factory + cross-validation
# ---------------------------------------------------------------------------
def _make_clf(classifier: str, n_estimators: int, rf_min_samples_leaf: int,
              random_state: int):
    if classifier == 'logreg':
        return Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                       random_state=random_state)),
        ])
    if classifier == 'random_forest':
        return RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=rf_min_samples_leaf,
            class_weight='balanced', random_state=random_state, n_jobs=-1)
    raise ValueError(f'Unknown classifier {classifier!r} '
                     f'(use "logreg" or "random_forest").')


def _importance(model, feat_names: list) -> np.ndarray:
    """Per-feature importance: |standardized coef| for logreg, impurity
    importance for RF."""
    est = model.named_steps['clf'] if isinstance(model, Pipeline) else model
    if hasattr(est, 'feature_importances_'):
        return np.asarray(est.feature_importances_, float)
    if hasattr(est, 'coef_'):
        return np.abs(np.ravel(est.coef_))
    return np.full(len(feat_names), np.nan)


def _cv_run(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, bins: np.ndarray,
            classifier: str, cv_scheme: str, n_splits: int, n_estimators: int,
            rf_min_samples_leaf: int, random_state: int) -> dict:
    """Spatially independent CV. Returns per-fold metrics, pooled out-of-fold
    predictions (for ROC / confusion / per-bin), and mean feature importance."""
    n_groups = len(np.unique(groups))
    if cv_scheme == 'leave_one_tile_out':
        splitter = LeaveOneGroupOut()
        n_eff = n_groups
    else:
        n_eff = min(n_splits, n_groups)
        splitter = GroupKFold(n_splits=n_eff)

    per_fold = []
    oof_prob = np.full(len(y), np.nan)
    imps = []
    feat_names = list(X.columns)
    Xv = X.to_numpy()
    for tr, te in splitter.split(Xv, y, groups):
        if len(np.unique(y[tr])) < 2:        # degenerate train fold
            continue
        model = _make_clf(classifier, n_estimators, rf_min_samples_leaf,
                          random_state)
        model.fit(Xv[tr], y[tr])
        prob = model.predict_proba(Xv[te])[:, 1]
        oof_prob[te] = prob
        pred = (prob >= 0.5).astype(int)
        fold = {'n_test': int(len(te)),
                'ba': float(balanced_accuracy_score(y[te], pred)),
                'f1': float(f1_score(y[te], pred, zero_division=0))}
        fold['auc'] = (float(roc_auc_score(y[te], prob))
                       if len(np.unique(y[te])) == 2 else np.nan)
        per_fold.append(fold)
        imps.append(_importance(model, feat_names))

    valid = ~np.isnan(oof_prob)
    oof_pred = np.where(oof_prob >= 0.5, 1, 0)
    return {
        'per_fold': per_fold,
        'n_folds': len(per_fold),
        'n_groups': n_groups,
        'cv_eff_splits': n_eff,
        'feat_names': feat_names,
        'mean_importance': (np.nanmean(imps, axis=0) if imps
                            else np.full(len(feat_names), np.nan)),
        'oof_prob': oof_prob,
        'oof_pred': oof_pred,
        'oof_valid': valid,
        'y': y,
        'bins': bins,
    }


def _agg(per_fold: list, key: str) -> tuple:
    vals = np.array([f[key] for f in per_fold], float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(vals.mean()), float(vals.std())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
_FSET_COLOR = {'height_only': 'C7', 'shape_only': 'C0',
               'height_shape': 'C3', 'extended': 'C2'}


def _plot_roc(results: dict, sensor: str, def_tag: str, save_path: Path) -> None:
    """Pooled out-of-fold ROC, one line per feature set, for one sensor."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['square'])
    ax.plot([0, 1], [0, 1], '--', color='k', lw=1, alpha=0.6)
    for fset, r in results.items():
        v = r['oof_valid']
        y, p = r['y'][v], r['oof_prob'][v]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=_FSET_COLOR.get(fset, None), lw=1.8,
                label=f'{fset} (AUC={auc:.3f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('False positive rate', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('True positive rate', fontsize=FONT_SIZES['label'])
    ax.set_title(f'{sensor.upper()} ROC ({def_tag})', fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'], loc='lower right')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_importance(r: dict, sensor: str, fset: str, def_tag: str,
                     save_path: Path) -> None:
    names = r['feat_names']
    imp = r['mean_importance']
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['small'])
    ax.barh(np.arange(len(names)), imp[order], color=_FSET_COLOR.get(fset, 'C0'),
            alpha=0.85)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([names[i] for i in order], fontsize=FONT_SIZES['ticks'])
    ax.set_xlabel('mean importance across folds', fontsize=FONT_SIZES['ticks'])
    ax.set_title(f'{sensor.upper()} {fset} ({def_tag})',
                 fontsize=FONT_SIZES['title'])
    ax.grid(True, axis='x', ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_confmat(r: dict, sensor: str, fset: str, def_tag: str,
                  save_path: Path) -> None:
    v = r['oof_valid']
    cm = confusion_matrix(r['y'][v], r['oof_pred'][v], labels=[0, 1])
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['mini'])
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=FONT_SIZES['label'])
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['sparse', 'dense'], fontsize=FONT_SIZES['ticks'])
    ax.set_yticklabels(['sparse', 'dense'], fontsize=FONT_SIZES['ticks'])
    ax.set_xlabel('predicted', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('true', fontsize=FONT_SIZES['label'])
    ax.set_title(f'{sensor.upper()} {fset}\n({def_tag})',
                 fontsize=FONT_SIZES['ticks'])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _perbin_table(results: dict, sensor: str, min_n: int) -> pd.DataFrame:
    """Out-of-fold AUC/BA/F1 within each source height bin, per feature set."""
    rows = []
    for fset, r in results.items():
        v = r['oof_valid']
        y, p, pred, bins = (r['y'][v], r['oof_prob'][v], r['oof_pred'][v],
                            r['bins'][v])
        for b in pd.unique(bins):
            mb = bins == b
            n = int(mb.sum())
            if n < min_n or len(np.unique(y[mb])) < 2:
                continue
            rows.append({'sensor': sensor, 'feature_set': fset, 'height_bin': b,
                         'n': n, 'auc': float(roc_auc_score(y[mb], p[mb])),
                         'ba': float(balanced_accuracy_score(y[mb], pred[mb])),
                         'f1': float(f1_score(y[mb], pred[mb], zero_division=0))})
    return pd.DataFrame(rows)


def _plot_perbin(perbin: pd.DataFrame, sensor: str, def_tag: str, metric: str,
                 save_path: Path) -> None:
    sub = perbin[perbin['sensor'] == sensor]
    if sub.empty:
        return
    bins = list(dict.fromkeys(sub['height_bin']))
    fsets = [f for f in ('height_only', 'shape_only', 'height_shape', 'extended')
             if f in set(sub['feature_set'])]
    x = np.arange(len(bins))
    w = 0.8 / max(len(fsets), 1)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
    for k, fset in enumerate(fsets):
        vals = [sub[(sub['height_bin'] == b) & (sub['feature_set'] == fset)]
                [metric].mean() for b in bins]
        ax.bar(x + (k - (len(fsets) - 1) / 2) * w, vals, w,
               label=fset, color=_FSET_COLOR.get(fset, None), alpha=0.85)
    ax.axhline(0.5, color='k', ls='--', lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=FONT_SIZES['ticks'])
    ax.set_xlabel('LVIS canopy-top height bin RH98 (m)',
                  fontsize=FONT_SIZES['label'])
    ax.set_ylabel(metric.upper(), fontsize=FONT_SIZES['label'])
    ax.set_title(f'{sensor.upper()} {metric.upper()} by height bin ({def_tag})',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'])
    ax.grid(True, axis='y', ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ===========================================================================
# Per-definition driver
# ===========================================================================
def _run_one_definition(df: pd.DataFrame, num_level: int, sensors: tuple,
                        extra_ok: list, height_edges: tuple, q: float,
                        min_bin_n: int, classifier: str, cv_scheme: str,
                        n_splits: int, n_estimators: int, rf_min_samples_leaf: int,
                        perbin_min_n: int, random_state: int, save_dir: Path,
                        say) -> tuple:
    def_tag = f'RH{num_level}/RH98'
    safe_tag = f'sub{num_level}'
    say('')
    say('=' * 74)
    say(f'DEFINITION: LVIS_sub = {def_tag}')
    say('=' * 74)

    labelled = _make_labels(df, num_level, height_edges, q, min_bin_n, say)
    if labelled.empty or labelled['tile'].nunique() < 2:
        say('  Too few labelled footprints / tiles for CV -> definition skipped.')
        return pd.DataFrame(), pd.DataFrame(), {}

    y = labelled['label'].to_numpy()
    groups = labelled['tile'].to_numpy()
    bins = labelled['height_bin'].to_numpy()

    summary_rows, perbin_all = [], []
    sensor_results = {}
    for sensor in sensors:
        say('')
        say(f'--- {sensor.upper()} ---')
        feats = _build_features(labelled, sensor, extra_ok)
        results = {}
        for fset, X in feats.items():
            r = _cv_run(X, y, groups, bins, classifier, cv_scheme, n_splits,
                        n_estimators, rf_min_samples_leaf, random_state)
            results[fset] = r
            auc_m, auc_s = _agg(r['per_fold'], 'auc')
            ba_m, ba_s = _agg(r['per_fold'], 'ba')
            f1_m, f1_s = _agg(r['per_fold'], 'f1')
            say(f'  {fset:<13} [{len(X.columns)} feat, {r["n_folds"]} folds] '
                f'AUC={auc_m:.3f}+/-{auc_s:.3f}  '
                f'BA={ba_m:.3f}+/-{ba_s:.3f}  F1={f1_m:.3f}+/-{f1_s:.3f}')
            summary_rows.append({
                'definition': def_tag, 'sensor': sensor.upper(),
                'feature_set': fset, 'n_features': len(X.columns),
                'n_folds': r['n_folds'], 'auc_mean': auc_m, 'auc_std': auc_s,
                'ba_mean': ba_m, 'ba_std': ba_s, 'f1_mean': f1_m,
                'f1_std': f1_s})
            # Deliverables: feature importance + confusion matrix per model.
            if len(X.columns) > 1:
                _plot_importance(r, sensor, fset, def_tag,
                                 save_dir / f'featimp_{safe_tag}_{sensor}_{fset}.png')
            _plot_confmat(r, sensor, fset, def_tag,
                          save_dir / f'confmat_{safe_tag}_{sensor}_{fset}.png')
        # ROC overlay + per-bin
        _plot_roc(results, sensor, def_tag,
                  save_dir / f'roc_{safe_tag}_{sensor}.png')
        pb = _perbin_table(results, sensor, perbin_min_n)
        perbin_all.append(pb)
        _plot_perbin(pb, sensor, def_tag, 'auc',
                     save_dir / f'perbin_{safe_tag}_{sensor}_auc.png')
        sensor_results[sensor] = results

    summary = pd.DataFrame(summary_rows)
    perbin = pd.concat(perbin_all, ignore_index=True) if perbin_all else pd.DataFrame()

    # Headline comparison: Delta vs height-only, per sensor.
    say('')
    say(f'Headline (definition {def_tag}): height_shape vs height_only')
    comp_rows = []
    for sensor in sensors:
        s = summary[summary['sensor'] == sensor.upper()].set_index('feature_set')
        if not {'height_only', 'height_shape'}.issubset(s.index):
            continue
        auc_h = s.loc['height_only', 'auc_mean']
        auc_hs = s.loc['height_shape', 'auc_mean']
        auc_sh = s.loc['shape_only', 'auc_mean'] if 'shape_only' in s.index else np.nan
        d_auc = auc_hs - auc_h
        d_ba = s.loc['height_shape', 'ba_mean'] - s.loc['height_only', 'ba_mean']
        shape_gain = auc_sh - auc_h
        comp_rows.append({'definition': def_tag, 'sensor': sensor.upper(),
                          'auc_height_only': auc_h, 'auc_shape_only': auc_sh,
                          'auc_height_shape': auc_hs,
                          'delta_auc': d_auc, 'delta_ba': d_ba,
                          'shape_minus_height_auc': shape_gain})
        say(f'  {sensor.upper():<5} AUC: height={s.loc["height_only","auc_mean"]:.3f} '
            f'shape={s.loc["shape_only","auc_mean"]:.3f} '
            f'height+shape={s.loc["height_shape","auc_mean"]:.3f} | '
            f'Delta_AUC={d_auc:+.3f}  Delta_BA={d_ba:+.3f}')
    comparison = pd.DataFrame(comp_rows)
    return summary, perbin, comparison


# ===========================================================================
# Interpretation
# ===========================================================================
def _interpret(comparison: pd.DataFrame, sensors: tuple, delta_auc_thresh: float,
               say) -> None:
    say('')
    say('=' * 74)
    say('INTERPRETATION — can GEDI / VSM separate LVIS dense vs sparse '
        'sub-canopy after controlling canopy-top height?')
    say('=' * 74)
    if comparison.empty:
        say('No comparison rows produced.')
        return
    for _, row in comparison.iterrows():
        improves = (np.isfinite(row['delta_auc']) and
                    row['delta_auc'] >= delta_auc_thresh)
        shape_alone = (np.isfinite(row['shape_minus_height_auc']) and
                       row['shape_minus_height_auc'] >= delta_auc_thresh)
        verdict = ('contains sub-canopy info beyond height'
                   if (improves and shape_alone) else
                   'weak/mixed evidence' if (improves or shape_alone) else
                   'little evidence of independent sub-canopy info')
        say(f'  [{row["definition"]}] {row["sensor"]}: '
            f'Delta_AUC={row["delta_auc"]:+.3f}, '
            f'shape-only gain={row["shape_minus_height_auc"]:+.3f} -> {verdict}.')

    # GEDI vs VSM signal strength (mean Delta_AUC across definitions).
    say('')
    say('GEDI vs VSM signal strength (mean Delta_AUC across definitions):')
    means = comparison.groupby('sensor')['delta_auc'].mean()
    for s in means.index:
        say(f'  {s}: mean Delta_AUC = {means[s]:+.3f}')
    if {'GEDI', 'VSM'}.issubset(means.index):
        g, v = means['GEDI'], means['VSM']
        if g >= delta_auc_thresh and v >= delta_auc_thresh:
            concl = ('Both GEDI and VSM improve over RH98-only -> both carry '
                     'information about lower-canopy structure beyond height.')
        elif g >= delta_auc_thresh and v < delta_auc_thresh:
            concl = ('GEDI shows sub-canopy signal that is only weakly preserved '
                     'in VSM.')
        elif v >= delta_auc_thresh and g < delta_auc_thresh:
            concl = ('VSM shows sub-canopy signal not matched by GEDI here '
                     '(unexpected — inspect data).')
        else:
            concl = ('Neither sensor improves much over RH98-only -> little '
                     'evidence of independent lower-canopy information.')
        say(f'  => {concl}')


# ===========================================================================
# Main entrypoint (registered as run=subcanopy_classify)
# ===========================================================================
def subcanopy_classification_analysis(
        pairs_dir: str,
        save_dir: str,
        pairs_glob: str = '*.parquet',
        sub_definitions: tuple = _DEFAULT_DEFS,
        height_bin_edges: tuple = (10.0, 20.0, 30.0, 40.0, 50.0),
        tercile_q: float = 0.30,
        min_bin_n: int = 100,
        rh98_min: float = 5.0,
        classifier: str = 'random_forest',
        cv_scheme: str = 'group_kfold',
        n_splits: int = 5,
        n_estimators: int = 300,
        rf_min_samples_leaf: int = 20,
        perbin_min_n: int = 60,
        random_state: int = 0,
        **kwargs) -> None:
    """Classify LVIS-defined dense vs sparse lower-canopy structure from GEDI and
    VSM RH features, after controlling canopy-top height via within-RH98-bin
    labelling and a held-out spatial split.

    Args:
        pairs_dir: dir of per-tile LVIS-GEDI-VSM pair parquets carrying
            `lvis_RH<NN>` / `vsm_RH<NN>` (uppercase) and `gedi_rh<NN>`
            (lowercase). Optional `<sensor>_fhd/_enl/_cr` enable the extended
            model when present on all sensors.
        save_dir: output dir for figures, CSVs and conclusion.md.
        pairs_glob: glob under pairs_dir. Default '*.parquet'.
        sub_definitions: LVIS RH numerators for LVIS_sub = RH<num>/RH98; the
            whole analysis is repeated per entry. Default (25, 50).
        height_bin_edges: lower edges of the LVIS RH98 height bins; last opens to
            +inf. Default (10,20,30,40,50) -> 10-20,20-30,30-40,40-50,50+.
        tercile_q: within-bin quantile for the dense/sparse split. Default 0.30
            (top 30% dense, bottom 30% sparse, middle 40% discarded).
        min_bin_n: drop a height bin with fewer than this many footprints before
            tercile labelling. Default 100.
        rh98_min: keep footprints with ALL sensors' RH98 > this (m). Default 5.
        classifier: 'random_forest' or 'logreg'. The SAME classifier is used for
            every feature set / sensor. Default 'random_forest'.
        cv_scheme: 'group_kfold' (GroupKFold by tile) or 'leave_one_tile_out'
            (LeaveOneGroupOut). Train and test footprints never share a tile.
        n_splits: folds for GroupKFold (capped at the number of tiles). Ignored
            for leave_one_tile_out. Default 5.
        n_estimators: trees for the random_forest classifier. Default 300.
        rf_min_samples_leaf: min samples per leaf for the random_forest. Default
            20.
        perbin_min_n: min out-of-fold footprints (both classes present) for a
            per-height-bin metric to be reported. Default 60.
        random_state: RNG seed for the classifiers.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    report: list = []

    def say(line: str = '') -> None:
        print(line)
        report.append(line)

    sensors = ('gedi', 'vsm')
    sub_definitions = tuple(int(d) for d in sub_definitions)
    height_bin_edges = tuple(float(e) for e in height_bin_edges)
    delta_auc_thresh = 0.02   # "meaningful" AUC improvement over height-only

    say('=' * 74)
    say('Sub-canopy classification — GEDI/VSM vs LVIS dense/sparse lower canopy')
    say('=' * 74)
    say(f'Classifier={classifier}; CV={cv_scheme} (n_splits={n_splits}); '
        f'tercile_q={tercile_q:g}; rh98_min={rh98_min:g} m; seed={random_state}.')
    say(f'Sub-canopy definitions: '
        + ', '.join(f'RH{d}/RH98' for d in sub_definitions) + '.')

    df, extra_ok = _load_pairs(pairs_dir, pairs_glob, sensors, sub_definitions, say)
    df = _clean(df, sensors, sub_definitions, rh98_min, say)

    all_summary, all_perbin, all_comp = [], [], []
    for num_level in sub_definitions:
        summary, perbin, comparison = _run_one_definition(
            df, num_level, sensors, extra_ok, height_bin_edges, tercile_q,
            min_bin_n, classifier, cv_scheme, n_splits, n_estimators,
            rf_min_samples_leaf, perbin_min_n, random_state, save_dir, say)
        if not summary.empty:
            all_summary.append(summary)
        if not perbin.empty:
            all_perbin.append(perbin)
        if not comparison.empty:
            all_comp.append(comparison)

    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    perbin = pd.concat(all_perbin, ignore_index=True) if all_perbin else pd.DataFrame()
    comparison = pd.concat(all_comp, ignore_index=True) if all_comp else pd.DataFrame()
    summary.to_csv(save_dir / 'summary_metrics.csv', index=False)
    perbin.to_csv(save_dir / 'per_height_bin_metrics.csv', index=False)
    comparison.to_csv(save_dir / 'comparison.csv', index=False)

    # Deliverable 1: a readable summary table in the report.
    if not summary.empty:
        say('')
        say('SUMMARY TABLE (mean across folds)')
        say('| Definition | Sensor | Features | AUC | Balanced Acc | F1 |')
        say('|---|---|---|---|---|---|')
        for _, r in summary.iterrows():
            say(f'| {r["definition"]} | {r["sensor"]} | {r["feature_set"]} '
                f'| {r["auc_mean"]:.3f}±{r["auc_std"]:.3f} '
                f'| {r["ba_mean"]:.3f}±{r["ba_std"]:.3f} '
                f'| {r["f1_mean"]:.3f}±{r["f1_std"]:.3f} |')

    _interpret(comparison, sensors, delta_auc_thresh, say)

    say('')
    say('Outputs: summary_metrics.csv, per_height_bin_metrics.csv, '
        'comparison.csv, roc_*.png, featimp_*.png, confmat_*.png, '
        'perbin_*_auc.png, conclusion.md')
    (save_dir / 'conclusion.md').write_text('\n'.join(report) + '\n')
    print(f'\n-> {save_dir / "conclusion.md"}')
