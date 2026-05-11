# -*- coding: utf-8 -*-
"""
Mesothelioma Xenium Interactive Explorer — SPATIAL135496 (GSM9654051)
- True transcript-level spatial dots (x_location, y_location, qv >= 20)
- Structure contours as polygon outlines (4 structures: 1-4 from histoseg)
- Toggle: % cells expressing  /  transcripts / cell (all)  /  transcripts / cell (expr+)
- Per-structure bar chart + inward density curve
Run: python meso_explorer_135496.py  ->  http://127.0.0.1:8052
"""
import io
import gzip
import zipfile
import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = r"Y:\long\publication_datasets\mesothelioma\SPATIAL135496"
TX_FILE    = rf"{BASE_DIR}\GSM9654051_SPATIAL135496_transcripts.parquet.gz"
HIST_ZIP   = rf"{BASE_DIR}\histoseg_outputs (15).zip"

QV_MIN = 20

STRUCT_COLORS = {
    'Structure 1': '#e41a1c',
    'Structure 2': '#377eb8',
    'Structure 3': '#4daf4a',
    'Structure 4': '#ff7f00',
}
STRUCTS = list(STRUCT_COLORS.keys())

# ── 1. Load histoseg partition ─────────────────────────────────────────────
print("Loading histoseg partition ...")
with zipfile.ZipFile(HIST_ZIP) as z:
    with z.open('cells_with_structure_partition.parquet') as f:
        part_df = pd.read_parquet(io.BytesIO(f.read()))

part_df = part_df.set_index('cell_id')
print(f"  {len(part_df):,} cells in partition")
for sname in STRUCTS:
    n = (part_df['isoline_structure_name'] == sname).sum()
    print(f"    {sname}: {n:,} cells")

# ── 2. Load contours & build KDTrees ──────────────────────────────────────
print("Loading contours ...")
contours = {s: [] for s in STRUCT_COLORS}

with zipfile.ZipFile(HIST_ZIP) as z:
    npy_files = sorted([n for n in z.namelist()
                        if re.match(r'structure_\d+_contour_\d+\.npy$', n)])
    for fname in npy_files:
        m = re.match(r'structure_(\d+)_contour_', fname)
        if not m:
            continue
        sname = f"Structure {m.group(1)}"
        if sname not in contours:
            continue
        with z.open(fname) as f:
            contours[sname].append(np.load(io.BytesIO(f.read())))

print("Building KDTrees & computing cell boundary distances ...")
kdtrees = {}
for sname, polys in contours.items():
    if not polys:
        continue
    all_pts = np.vstack(polys)
    kdtrees[sname] = cKDTree(all_pts)
    print(f"  {sname}: {len(polys)} contours, {len(all_pts):,} boundary pts")

dist_arr = np.full(len(part_df), np.nan)
for sname, tree in kdtrees.items():
    mask = (part_df['isoline_structure_name'] == sname).values
    if mask.sum() == 0:
        continue
    pts = part_df[mask][['x_centroid', 'y_centroid']].values
    d, _ = tree.query(pts, workers=-1)
    dist_arr[mask] = d
    print(f"  {sname}: {mask.sum():,} cells  max={d.max():.1f} um  median={np.median(d):.1f} um")

part_df['dist_to_boundary'] = dist_arr

# ── 3. Static contour traces ───────────────────────────────────────────────
def make_contour_traces():
    traces = []
    for sname, color in STRUCT_COLORS.items():
        polys = contours.get(sname, [])
        if not polys:
            continue
        xs, ys = [], []
        for poly in polys:
            xs.extend(list(poly[:, 0]) + [poly[0, 0], np.nan])
            ys.extend(list(poly[:, 1]) + [poly[0, 1], np.nan])
        traces.append(go.Scattergl(
            x=xs, y=[-v for v in ys],
            mode='lines',
            line=dict(color=color, width=1.2),
            name=sname, legendgroup=sname,
            hoverinfo='skip',
        ))
    return traces

CONTOUR_TRACES = make_contour_traces()

# ── 4. Load & annotate transcripts ────────────────────────────────────────
print("Loading transcripts ...")
with gzip.open(TX_FILE, 'rb') as f:
    tx_raw = pd.read_parquet(io.BytesIO(f.read()),
                             columns=['feature_name', 'x_location', 'y_location', 'qv', 'cell_id'])

tx = tx_raw[tx_raw['qv'] >= QV_MIN].copy()
del tx_raw
print(f"  Transcripts after qv>={QV_MIN}: {len(tx):,}")

tx['structure'] = tx['cell_id'].map(part_df['isoline_structure_name'])
tx['dist']      = tx['cell_id'].map(part_df['dist_to_boundary'])
tx = tx[tx['structure'].notna()].copy()
print(f"  Transcripts with structure: {len(tx):,}")

print("Building per-gene index ...")
GENE_TX = {}
for gene, grp in tx.groupby('feature_name', observed=True):
    GENE_TX[gene] = grp[['x_location', 'y_location', 'structure', 'dist', 'cell_id']].copy()
GENES = sorted(GENE_TX.keys())
print(f"  {len(GENES)} genes indexed.")

STRUCT_NCELLS = part_df.groupby('isoline_structure_name', observed=True).size().to_dict()

def get_cells_in_bin(sname, lo, hi):
    mask = (
        (part_df['isoline_structure_name'] == sname) &
        (part_df['dist_to_boundary'] >= lo) &
        (part_df['dist_to_boundary'] < hi) &
        np.isfinite(part_df['dist_to_boundary'])
    )
    return part_df.index[mask]

print(f"Ready. Starting Dash ...")

# ── Metric configs ─────────────────────────────────────────────────────────
METRIC_OPTIONS = [
    {'label': '% cells expressing',             'value': 'pct'},
    {'label': 'Transcripts / cell (all cells)',  'value': 'mean_all'},
    {'label': 'Transcripts / cell (expr+ only)', 'value': 'mean_expr'},
]
METRIC_YLAB = {
    'pct':       '% cells expressing',
    'mean_all':  'Transcripts per cell (all)',
    'mean_expr': 'Transcripts per cell (expr+)',
}
METRIC_BAR_FMT = {
    'pct':       lambda v: f'{v:.1f}%',
    'mean_all':  lambda v: f'{v:.3f}',
    'mean_expr': lambda v: f'{v:.2f}',
}

N_BINS = 20

def per_structure_bar(gene_df, metric):
    vals = []
    for sname in STRUCTS:
        sub     = gene_df[gene_df['structure'] == sname]
        n_cells = STRUCT_NCELLS.get(sname, 1)
        n_tx    = len(sub)
        if metric == 'pct':
            vals.append(100.0 * sub['cell_id'].nunique() / max(n_cells, 1))
        elif metric == 'mean_all':
            vals.append(n_tx / max(n_cells, 1))
        else:
            n_expr = sub['cell_id'].nunique()
            vals.append(n_tx / max(n_expr, 1))
    return vals

def distance_curve(gene_df, sname, metric='pct', max_dist=200, n_bins=N_BINS):
    sub_s = gene_df[
        (gene_df['structure'] == sname) &
        np.isfinite(gene_df['dist']) &
        (gene_df['dist'] <= max_dist)
    ]
    if len(sub_s) < 5:
        return None, None

    bins = np.linspace(0, max_dist, n_bins + 1)
    vals, mids = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        bin_cell_ids = get_cells_in_bin(sname, lo, hi)
        n_cells = len(bin_cell_ids)
        if n_cells < 5:
            continue

        tx_bin = sub_s[(sub_s['dist'] >= lo) & (sub_s['dist'] < hi)]
        n_tx   = len(tx_bin)

        if metric == 'pct':
            expr_set = set(tx_bin['cell_id'].values) & set(bin_cell_ids)
            vals.append(100.0 * len(expr_set) / n_cells)
        elif metric == 'mean_all':
            vals.append(n_tx / n_cells)
        else:
            n_expr = tx_bin['cell_id'].nunique()
            vals.append(n_tx / max(n_expr, 1))

        mids.append((lo + hi) / 2)

    return (np.array(mids), np.array(vals)) if vals else (None, None)

# ── App layout ─────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title='Meso Explorer SPATIAL135496')

app.layout = html.Div([

    html.Div([
        html.H2('Mesothelioma Xenium — SPATIAL135496 (GSM9654051)',
                style={'margin': '0', 'fontSize': '17px', 'whiteSpace': 'nowrap'}),
        html.Div([
            html.Label('Gene:', style={'fontWeight': 'bold', 'marginRight': '6px'}),
            dcc.Dropdown(
                id='gene-dd',
                options=[{'label': g, 'value': g} for g in GENES],
                value='MSLN', searchable=True, clearable=False,
                style={'width': '220px'},
            ),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Label('Metric:', style={'fontWeight': 'bold', 'marginRight': '6px'}),
            dcc.RadioItems(
                id='metric-radio',
                options=METRIC_OPTIONS,
                value='pct',
                inline=True,
                inputStyle={'marginRight': '4px'},
                labelStyle={'marginRight': '16px', 'fontSize': '13px'},
            ),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Label('Max dist (um):', style={'marginRight': '4px', 'fontSize': '13px'}),
            dcc.Input(id='max-dist', type='number', value=200,
                      min=10, max=1000, step=10, style={'width': '60px'}),
            html.Label('  Sample/structure:', style={'marginLeft': '12px', 'marginRight': '4px', 'fontSize': '13px'}),
            dcc.Input(id='max-tx', type='number', value=50000,
                      min=1000, max=500000, step=5000, style={'width': '80px'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
    ], style={
        'background': '#f4f4f4', 'padding': '10px 18px',
        'borderBottom': '1px solid #ccc',
        'display': 'flex', 'alignItems': 'center',
        'gap': '24px', 'flexWrap': 'wrap',
    }),

    html.Div([
        html.Div([
            dcc.Graph(id='spatial-plot',
                      style={'height': 'calc(100vh - 88px)'},
                      config={'scrollZoom': True,
                              'toImageButtonOptions': {
                                  'format': 'png', 'filename': 'meso135496_spatial',
                                  'height': 1400, 'width': 1200, 'scale': 2}}),
        ], style={'width': '57%', 'borderRight': '1px solid #ddd'}),

        html.Div([
            dcc.Graph(id='stats-plot',
                      style={'height': 'calc(100vh - 88px)'},
                      config={'toImageButtonOptions': {
                          'format': 'png', 'filename': 'meso135496_stats', 'scale': 2}}),
        ], style={'width': '43%'}),

    ], style={'display': 'flex'}),

], style={'fontFamily': 'Arial', 'margin': 0, 'padding': 0})


@app.callback(
    Output('spatial-plot', 'figure'),
    Output('stats-plot', 'figure'),
    Input('gene-dd', 'value'),
    Input('metric-radio', 'value'),
    Input('max-dist', 'value'),
    Input('max-tx', 'value'),
)
def update(gene, metric, max_dist, max_tx):
    max_dist = float(max_dist or 200)
    max_tx   = int(max_tx or 50000)
    ylab     = METRIC_YLAB[metric]
    fmt      = METRIC_BAR_FMT[metric]
    gene_df  = GENE_TX.get(gene)

    # ── Spatial ───────────────────────────────────────────────────────────
    fig_s = go.Figure()
    for tr in CONTOUR_TRACES:
        fig_s.add_trace(tr)

    n_total = 0
    if gene_df is not None:
        n_total = len(gene_df)
        for sname, color in STRUCT_COLORS.items():
            sub = gene_df[gene_df['structure'] == sname]
            if len(sub) == 0:
                continue
            if len(sub) > max_tx:
                sub = sub.sample(max_tx, random_state=42)
            fig_s.add_trace(go.Scattergl(
                x=sub['x_location'].values,
                y=-sub['y_location'].values,
                mode='markers',
                marker=dict(size=1.8, color=color, opacity=0.55, line=dict(width=0)),
                name=sname, legendgroup=sname,
                showlegend=False,
                hovertemplate=f'<b>{sname}</b><extra></extra>',
            ))

    sample_note = f'  [max {max_tx:,}/structure]' if max_tx < 300000 else ''
    fig_s.update_layout(
        title=dict(
            text=f'<b>{gene}</b>  |  {n_total:,} transcripts (qv>={QV_MIN}){sample_note}',
            x=0.5, font_size=12),
        plot_bgcolor='#181818', paper_bgcolor='white',
        xaxis=dict(visible=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(visible=False),
        legend=dict(title='Structure', font_size=10,
                    bgcolor='rgba(255,255,255,0.85)', itemsizing='constant'),
        margin=dict(l=5, r=5, t=38, b=5),
        uirevision='spatial',
    )

    # ── Stats ─────────────────────────────────────────────────────────────
    fig_r = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f'<b>{gene}</b> per structure  —  {ylab}',
            f'Inward density  (0 = boundary  ->  {int(max_dist)} um inside)',
        ],
        vertical_spacing=0.13,
        row_heights=[0.36, 0.64],
    )

    if gene_df is not None:
        bar_vals = per_structure_bar(gene_df, metric)
        colors   = list(STRUCT_COLORS.values())

        fig_r.add_trace(go.Bar(
            x=STRUCTS, y=bar_vals,
            marker_color=colors,
            text=[fmt(v) for v in bar_vals],
            textposition='outside',
            hovertemplate='%{x}<br>' + ylab + ': %{y:.3f}<extra></extra>',
            showlegend=False,
        ), row=1, col=1)

        y_max = max(bar_vals) if bar_vals else 1
        fig_r.update_yaxes(title_text=ylab, range=[0, y_max * 1.25 + 1e-9], row=1, col=1)
        fig_r.update_xaxes(tickangle=-30, row=1, col=1)

        for sname, color in STRUCT_COLORS.items():
            mids, curve_vals = distance_curve(gene_df, sname,
                                              metric=metric, max_dist=max_dist)
            if mids is None:
                continue
            fig_r.add_trace(go.Scatter(
                x=mids, y=curve_vals,
                mode='lines+markers',
                name=sname,
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=(
                    f'<b>{sname}</b><br>'
                    'Distance: %{x:.0f} um<br>'
                    + ylab + ': %{y:.3f}<extra></extra>'),
                legendgroup=sname,
                showlegend=True,
            ), row=2, col=1)

        fig_r.update_xaxes(title_text='Distance from boundary (um)', row=2, col=1)
        fig_r.update_yaxes(title_text=ylab, row=2, col=1)

    fig_r.update_layout(
        paper_bgcolor='white', plot_bgcolor='#f9f9f9',
        margin=dict(l=55, r=15, t=65, b=40),
        legend=dict(title='Structure', font_size=11, x=0.76, y=0.33),
        uirevision='stats',
    )

    return fig_s, fig_r


if __name__ == '__main__':
    app.run(debug=False, port=8052)
