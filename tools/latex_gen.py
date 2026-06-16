from pathlib import Path
from typing import Dict, List
import pandas as pd


def csv_to_latex_rh(csv_file: str, precision: int = 2, **kwargs):
    csv_file = Path(csv_file).expanduser()
    output_file = csv_file.with_suffix('.tex')
    df = pd.read_csv(csv_file, index_col=0)
    df.index = df.index.str.split('_', expand=True, n=1)
    df = df.unstack(level=0)
    ordered_cols = df.columns.get_level_values(1).unique()
    ordered_cols = sorted(ordered_cols, key=lambda x: int(x.split('RH')[1]))
    df = df.reindex(columns=ordered_cols, level=1)
    n_cols = df.shape[1]
    df.to_latex(str(output_file), float_format=f"%.{precision}f", na_rep='', column_format=f'l *{{{n_cols}}}{{S}}')
    return df


def compare_csv_to_latex(
    csv_files: Dict[str, str],
    output_file: str = None,
    precision: int = 2,
    index_col=None,
    drop_cols: List[str] = None,
    metric_order: List[str] = None,
    rename_metrics: Dict[str, str] = None,
    row_labels: List[str] = None,
    **kwargs,
):
    """Combine several same-shaped stats tables into one LaTeX table with a
    two-level header: outer = metric (R2, RMSE, ...), inner = source (VSM, GEDI).

    Args:
        csv_files: mapping {source_label: csv_path}, e.g.
            {VSM: .../overall_stats_vsm.csv, GEDI: .../overall_stats_gedi.csv}.
            All CSVs must share the same columns (the metrics).
        output_file: .tex path; defaults to <first_csv_dir>/compare_<first_stem>.tex.
        index_col: passed to pd.read_csv. Use None for single-row stats tables
            (no index column), or 0 when the first column holds row labels.
        drop_cols: metric columns to drop before merging (e.g. [n]).
        metric_order: order/subset of metrics for the outer header (post-rename).
        rename_metrics: {csv_col: display_name}, e.g. {r2: R$^2$, me: ME (bias)}.
        row_labels: replace the row index labels (e.g. [LVIS]).
    """
    sources = list(csv_files.keys())
    dfs = {}
    first_path = None
    for label, path in csv_files.items():
        path = Path(path).expanduser()
        first_path = first_path or path
        df = pd.read_csv(path, index_col=index_col)
        if drop_cols is not None:
            df = df.drop(columns=drop_cols)
        if rename_metrics is not None:
            df = df.rename(columns=rename_metrics)
        dfs[label] = df

    # MultiIndex columns: (source, metric) -> swap to (metric, source)
    combined = pd.concat(dfs, axis=1).swaplevel(0, 1, axis=1)

    metrics = metric_order if metric_order is not None else list(dfs[sources[0]].columns)
    combined = combined.reindex(columns=pd.MultiIndex.from_product([metrics, sources]))

    if row_labels is not None:
        combined.index = row_labels

    # Brace-wrap the inner source labels so siunitx treats them as text, not
    # numbers, in the S columns (else e.g. "GEDI" -> "Invalid number 'ED'").
    combined.columns = combined.columns.set_levels(
        [f'{{{s}}}' for s in combined.columns.levels[1]], level=1
    )

    if output_file is None:
        output_file = first_path.with_name(f'compare_{first_path.stem}.tex')
    output_file = Path(output_file).expanduser()

    n_cols = combined.shape[1]
    latex = combined.to_latex(
        None,
        float_format=f"%.{precision}f",
        na_rep='',
        multicolumn=True,
        multicolumn_format='c',
        column_format=f'l *{{{n_cols}}}{{S}}',
    )

    # Add a \cmidrule(lr) under each metric group (between the metric and the
    # source header rows). Each metric spans len(sources) cols, after the index.
    n_src = len(sources)
    cmid = ' '.join(
        f'\\cmidrule(lr){{{2 + i * n_src}-{1 + (i + 1) * n_src}}}'
        for i in range(len(metrics))
    )
    lines = latex.splitlines()
    for idx, line in enumerate(lines):
        if '\\multicolumn' in line:
            lines.insert(idx + 1, cmid)
            break
    latex = '\n'.join(lines) + '\n'
    output_file.write_text(latex)
    return combined


def csv_to_latex(csv_file: str, precision: int = 2, drop_cols: List[str] = None, **kwargs):
    csv_file = Path(csv_file).expanduser()
    output_file = csv_file.with_suffix('.tex')
    df = pd.read_csv(csv_file, index_col=0)
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)
    n_cols = df.shape[1]
    df.to_latex(str(output_file), float_format=f"%.{precision}f", na_rep='', column_format=f'l *{{{n_cols}}}{{S}}')
    return df
