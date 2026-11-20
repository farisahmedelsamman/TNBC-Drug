"""Ferroptosis drug combination prediction page — Streamlit components.

Renders the RSL3 + Erastin and RSL3 + Sulfasalazine combination prediction UI.
Imported by dashboard/app.py.

Scientific basis: Bliss Independence model applied to 4PL curves reconstructed
from CTRP-measured (RSL3, Erastin) and PRISM-estimated (Sulfasalazine) AUC values.
See src/models/combination_predictor.py for full methodology and limitations.
"""
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'models'))

try:
    from combination_predictor import (
        bliss_combination_surface,
        predict_single_drug,
        bootstrap_combination_summary,
        SUPPORTED_COMBINATIONS,
        SUPPORTED_CELL_LINES,
        CTRP_AUC,
        DATA_SOURCE,
        CONFIDENCE,
    )
    _MODULE_OK = True
    _MODULE_ERR = None
except Exception as e:
    _MODULE_OK = False
    _MODULE_ERR = str(e)


# ── Colour palette ──────────────────────────────────────────────────────────
DRUG_A_COLOR = '#D9480F'   # RSL3 — orange-red
DRUG_B_COLORS = {
    'ERASTIN': '#1c7ed6',      # blue
    'SULFASALAZINE': '#7950f2', # purple
}
COMBO_COLOR = '#2f7d32'    # dark green for combination

LIMITATION_COLOR = '#fff3cd'   # yellow-ish warning box


def _drug_b_color(drug_b: str) -> str:
    return DRUG_B_COLORS.get(drug_b, '#495057')


# ── Low-level chart helpers ─────────────────────────────────────────────────

def _dose_response_chart(result_a, result_b, combo_key: str):
    """Overlay drug A, drug B and Bliss combination single-agent curves."""
    drug_a = result_a['drug']
    drug_b = result_b['drug']
    color_b = _drug_b_color(drug_b)

    fig = go.Figure()

    # Drug A
    fig.add_trace(go.Scatter(
        x=result_a['conc_um'], y=result_a['viability'],
        mode='lines', name=f'{drug_a} alone',
        line=dict(color=DRUG_A_COLOR, width=2.5),
        hovertemplate=f'{drug_a}<br>Conc: %{{x:.3g}} µM<br>Viability: %{{y:.3f}}<extra></extra>',
    ))

    # Drug B
    fig.add_trace(go.Scatter(
        x=result_b['conc_um'], y=result_b['viability'],
        mode='lines', name=f'{drug_b} alone',
        line=dict(color=color_b, width=2.5),
        hovertemplate=f'{drug_b}<br>Conc: %{{x:.3g}} µM<br>Viability: %{{y:.3f}}<extra></extra>',
    ))

    # 50% viability reference
    fig.add_hline(y=0.5, line=dict(dash='dash', color='#868E96', width=1),
                  annotation_text='50% viability (IC50 reference)',
                  annotation_position='bottom right',
                  annotation_font=dict(color='#868E96', size=10))

    fig.update_layout(
        title=dict(text='Single-agent dose-response curves (4PL reconstruction from AUC)', font=dict(size=13)),
        xaxis=dict(type='log', title='Concentration (µM)', tickformat='.2g'),
        yaxis=dict(title='Predicted viability', range=[-0.05, 1.05]),
        legend=dict(x=0.02, y=0.05, bgcolor='rgba(0,0,0,0)'),
        height=380, margin=dict(t=44, b=40, l=10, r=10),
        hovermode='x unified',
    )
    return fig


def _combination_heatmap(combo_result, metric: str = 'effect'):
    """Heatmap of the Bliss combination response surface."""
    conc_a = combo_result['conc_a']
    conc_b = combo_result['conc_b']
    drug_a = combo_result['drug_a']
    drug_b = combo_result['drug_b']

    if metric == 'effect':
        values = combo_result['effect_combination']
        title = f'Bliss-predicted inhibition (effect): {drug_a} × {drug_b}'
        colorscale = 'RdYlGn_r'
        cbar_title = 'Effect (0=alive, 1=dead)'
        zmin, zmax = 0.0, 1.0
    else:
        values = combo_result['viab_combination']
        title = f'Bliss-predicted viability: {drug_a} × {drug_b}'
        colorscale = 'RdYlGn'
        cbar_title = 'Viability (1=alive, 0=dead)'
        zmin, zmax = 0.0, 1.0

    def _fmt(v):
        if v == 0:
            return '0'
        return f'{v:.2g}'

    row_labels = [_fmt(v) + ' µM' for v in conc_a]
    col_labels = [_fmt(v) + ' µM' for v in conc_b]

    df = pd.DataFrame(values, index=row_labels, columns=col_labels)

    fig = px.imshow(
        df, text_auto='.2f', aspect='auto',
        color_continuous_scale=colorscale,
        zmin=zmin, zmax=zmax,
    )
    fig.update_traces(textfont=dict(size=8))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis_title=f'{drug_b} concentration',
        yaxis_title=f'{drug_a} concentration',
        coloraxis_colorbar_title=cbar_title,
        height=430, margin=dict(t=44, b=40, l=10, r=10),
    )
    fig.update_xaxes(side='bottom')
    return fig


def _single_agent_effect_chart(combo_result):
    """Side-by-side dose-response comparison chart in effect space."""
    conc_a = combo_result['conc_a']
    conc_b = combo_result['conc_b']
    drug_a = combo_result['drug_a']
    drug_b = combo_result['drug_b']
    effect_a = combo_result['effect_a']
    effect_b = combo_result['effect_b']
    color_b = _drug_b_color(drug_b)

    # Equimolar combination curve
    n = max(len(conc_a), len(conc_b))
    all_c = np.unique(np.concatenate([conc_a[conc_a > 0], conc_b[conc_b > 0]]))
    if len(all_c) > 1:
        t = np.geomspace(all_c.min(), all_c.max(), 200)
    else:
        t = np.array([1.0])

    from combination_predictor import viability_from_params
    pa = combo_result['params_a']
    pb = combo_result['params_b']
    va_t = viability_from_params(t, *pa)
    vb_t = viability_from_params(t, *pb)
    combo_t = 1.0 - va_t * vb_t  # effect

    fig = go.Figure()
    nz_a = conc_a > 0
    nz_b = conc_b > 0

    fig.add_trace(go.Scatter(
        x=conc_a[nz_a], y=effect_a[nz_a],
        mode='lines', name=f'{drug_a} (effect)',
        line=dict(color=DRUG_A_COLOR, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=conc_b[nz_b], y=effect_b[nz_b],
        mode='lines', name=f'{drug_b} (effect)',
        line=dict(color=color_b, width=2.5, dash='dash'),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=combo_t,
        mode='lines', name='Bliss combination (equimolar)',
        line=dict(color=COMBO_COLOR, width=2.5, dash='dot'),
    ))
    fig.add_hline(y=0.5, line=dict(dash='dot', color='#868E96', width=1),
                  annotation_text='50% inhibition',
                  annotation_position='bottom right',
                  annotation_font=dict(color='#868E96', size=10))
    fig.update_layout(
        title=dict(text='Effect (1 − viability) vs concentration', font=dict(size=13)),
        xaxis=dict(type='log', title='Concentration (µM)', tickformat='.2g'),
        yaxis=dict(title='Fractional inhibition', range=[-0.03, 1.03]),
        legend=dict(x=0.02, y=0.95, bgcolor='rgba(0,0,0,0)'),
        height=380, margin=dict(t=44, b=40, l=10, r=10),
        hovermode='x unified',
    )
    return fig


def _ic50_summary_bars(pred_a, pred_b, combo_ic50):
    """Bar chart comparing individual IC50s and combination IC50."""
    rows = []
    if pred_a['ic50_um'] is not None:
        rows.append({'Drug': pred_a['drug'] + ' (single)', 'IC50 (µM)': pred_a['ic50_um'],
                     'color': DRUG_A_COLOR})
    else:
        rows.append({'Drug': pred_a['drug'] + ' (single)', 'IC50 (µM)': None, 'color': DRUG_A_COLOR})

    if pred_b['ic50_um'] is not None:
        rows.append({'Drug': pred_b['drug'] + ' (single)', 'IC50 (µM)': pred_b['ic50_um'],
                     'color': _drug_b_color(pred_b['drug'])})

    if combo_ic50.get('estimable'):
        rows.append({
            'Drug': f"{pred_a['drug']} + {pred_b['drug']}\n(Bliss 1:1 equimolar)",
            'IC50 (µM)': combo_ic50['ic50_each_drug_um'],
            'color': COMBO_COLOR,
        })

    df = pd.DataFrame([r for r in rows if r['IC50 (µM)'] is not None])
    if df.empty:
        return None

    fig = px.bar(
        df, x='Drug', y='IC50 (µM)',
        color='Drug',
        color_discrete_map={r['Drug']: r['color'] for r in rows if r['IC50 (µM)'] is not None},
        text='IC50 (µM)',
        labels={'IC50 (µM)': 'IC50 (µM) — lower = more potent'},
    )
    fig.update_traces(texttemplate='%{y:.3g} µM', textposition='outside')
    fig.update_layout(
        title=dict(text='IC50 comparison (4PL-predicted, model-based)', font=dict(size=12)),
        showlegend=False, height=350,
        margin=dict(t=44, b=50, l=10, r=10),
        yaxis_type='log',
    )
    return fig


# ── Main render function ────────────────────────────────────────────────────

def render_combination_prediction_tab(st):
    """Render the full combination prediction tab in a Streamlit app.

    Call this from the app.py tab rendering block:

        from combination_page import render_combination_prediction_tab
        render_combination_prediction_tab(st)
    """
    if not _MODULE_OK:
        st.error(f'Combination predictor module failed to load: {_MODULE_ERR}')
        return

    st.subheader('Ferroptosis Drug Combination Prediction (RSL3 + Erastin / Sulfasalazine)')

    # ── Scientific caveat banner ─────────────────────────────────────────────
    st.warning(
        '**⚠ Model-Based Estimates — Not Experimentally Validated**  \n'
        'All predictions on this page are derived from the **Bliss Independence null model** '
        'applied to single-agent 4PL curves reconstructed from published AUC values '
        '(CTRPv2.1 for RSL3 & Erastin; PRISM model for Sulfasalazine).  \n'
        'These values represent **additive reference predictions** — the expected combined '
        'response if the two drugs act independently.  '
        'They **cannot establish synergy**, which requires comparison against experimental '
        'checkerboard assay data.  '
        'See the **Synergy Checkerboard** tab to enter actual plate readings.'
    )

    # ── Controls ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        combo_key = st.selectbox(
            'Drug combination',
            list(SUPPORTED_COMBINATIONS.keys()),
            help='Select the two-drug combination to predict',
            key='combo_pred_pair',
        )
    with col2:
        cell_line = st.selectbox(
            'Cell line',
            list(SUPPORTED_CELL_LINES.keys()),
            format_func=lambda x: f'{x} ({SUPPORTED_CELL_LINES[x]["subtype"]})',
            help='Select the TNBC (or control) cell line',
            key='combo_pred_line',
        )

    combo_cfg = SUPPORTED_COMBINATIONS[combo_key]
    drug_a = combo_cfg['drug_a']   # always RSL3
    drug_b = combo_cfg['drug_b']

    # Concentration range controls
    st.markdown('**Concentration ranges (µM)**')
    c1, c2, c3, c4 = st.columns(4)
    default_ranges = {
        'RSL3': (0.01, 10.0),
        'ERASTIN': (0.1, 100.0),
        'SULFASALAZINE': (10.0, 2000.0),
    }
    dr_a = default_ranges.get(drug_a, (0.01, 100.0))
    dr_b = default_ranges.get(drug_b, (0.01, 100.0))

    min_a = c1.number_input(f'{drug_a} min (µM)', value=float(dr_a[0]), min_value=1e-6,
                              format='%g', key='combo_min_a')
    max_a = c2.number_input(f'{drug_a} max (µM)', value=float(dr_a[1]), min_value=1e-5,
                              format='%g', key='combo_max_a')
    min_b = c3.number_input(f'{drug_b} min (µM)', value=float(dr_b[0]), min_value=1e-6,
                              format='%g', key='combo_min_b')
    max_b = c4.number_input(f'{drug_b} max (µM)', value=float(dr_b[1]), min_value=1e-5,
                              format='%g', key='combo_max_b')

    n_grid = st.slider('Grid resolution (points per axis)', 8, 20, 12,
                        help='More points = smoother surface but slower render',
                        key='combo_n_grid')

    if min_a >= max_a or min_b >= max_b:
        st.error('Minimum concentration must be less than maximum.')
        return

    # ── Compute predictions ───────────────────────────────────────────────────
    try:
        pred_a = predict_single_drug(drug_a, cell_line,
                                      conc_range=(min_a, max_a), n_points=200)
        pred_b = predict_single_drug(drug_b, cell_line,
                                      conc_range=(min_b, max_b), n_points=200)
        combo = bliss_combination_surface(
            drug_a, drug_b, cell_line,
            conc_range_a=(min_a, max_a),
            conc_range_b=(min_b, max_b),
            n_points=n_grid + 1,  # +1 because we add zero
        )
    except ValueError as e:
        st.error(f'Prediction error: {e}')
        return

    bs = combo['bliss_summary']
    ic50_combo = combo['ic50_combo']
    auc_sum = combo['auc_summary']

    # ── Metric cards ─────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('#### Predicted Response Metrics')

    m1, m2, m3, m4 = st.columns(4)
    # RSL3 IC50
    if pred_a['ic50_um'] is not None:
        m1.metric(
            f'{drug_a} IC50',
            f"{pred_a['ic50_um']:.3g} µM",
            help=f'4PL-predicted IC50 from CTRP AUC = {pred_a["auc_norm_ctrp"]:.4f}. '
                 f'Data: {pred_a["confidence"]}',
        )
    else:
        m1.metric(f'{drug_a} IC50', 'Not estimable',
                  help='Drug does not reach 50% inhibition in this concentration range')

    # Drug B IC50
    if pred_b['ic50_um'] is not None:
        m2.metric(
            f'{drug_b} IC50',
            f"{pred_b['ic50_um']:.3g} µM",
            help=f'4PL-predicted IC50 from AUC = {pred_b["auc_norm_ctrp"]:.4f}. '
                 f'Data: {pred_b["confidence"]}',
        )
    else:
        m2.metric(f'{drug_b} IC50', 'Not estimable')

    # Combo IC50
    if ic50_combo.get('estimable'):
        m3.metric(
            'Combo IC50 (each drug, 1:1)',
            f"{ic50_combo['ic50_each_drug_um']:.3g} µM",
            help=(
                'IC50 along the equimolar (1:1) trajectory: concentration of each drug '
                f'at which V_AB = 0.5. Total = {ic50_combo["ic50_total_um"]:.3g} µM. '
                'Bliss-model estimate — not a measured value.'
            ),
        )
    else:
        m3.metric('Combo IC50 (1:1)', 'Not estimable in range',
                  help=ic50_combo.get('reason', ''))

    # Bliss expected top-dose effect
    m4.metric(
        'Bliss E_AB (top doses)',
        f"{bs['bliss_expected_top']:.3f}",
        help=(
            'Expected combined inhibition at the highest dose of both drugs under Bliss '
            'independence (additive null hypothesis). NOT a synergy prediction.'
        ),
    )

    # AUC metrics
    m5, m6, m7, m8 = st.columns(4)
    m5.metric(
        f'{drug_a} AUC (CTRP)',
        f"{pred_a['auc_norm_ctrp']:.4f}",
        help='Measured AUC from CTRPv2.1 (0-1 scale, lower = more sensitive). '
             + pred_a['confidence'],
    )
    m6.metric(
        f'{drug_b} AUC (source)',
        f"{pred_b['auc_norm_ctrp']:.4f}",
        help='Source AUC (0-1 scale, lower = more sensitive). ' + pred_b['confidence'],
    )
    m7.metric(
        'Single-agent AUC ratio',
        f"{pred_b['auc_norm_ctrp']/pred_a['auc_norm_ctrp']:.2f}×",
        help=f'{drug_b} AUC / {drug_a} AUC. Values >1 indicate {drug_a} is more potent.',
    )
    bliss_e_a = 1 - pred_a['auc_norm_ctrp']
    bliss_e_b = 1 - pred_b['auc_norm_ctrp']
    bliss_e_ab = bliss_e_a + bliss_e_b - bliss_e_a * bliss_e_b
    m8.metric(
        'Bliss E_AB (AUC-level)',
        f'{bliss_e_ab:.3f}',
        help=(
            f'Bliss additive effect using the source AUC values as single-point proxies: '
            f'E_A={bliss_e_a:.3f}, E_B={bliss_e_b:.3f}, E_AB=E_A+E_B-E_A×E_B={bliss_e_ab:.3f}. '
            'This is the same calculation used in the FerroSelect panel combo ranking.'
        ),
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('#### Dose-Response Curves')

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            _dose_response_chart(pred_a, pred_b, combo_key),
            use_container_width=True, config={'displaylogo': False},
        )
    with chart_col2:
        st.plotly_chart(
            _single_agent_effect_chart(combo),
            use_container_width=True, config={'displaylogo': False},
        )

    # ── Combination response surface ──────────────────────────────────────────
    st.markdown('---')
    st.markdown('#### Bliss Additive Combination Response Surface')
    st.caption(
        'The heatmap shows the **expected combined response under Bliss independence** '
        '(null hypothesis: two drugs act independently).  This is the reference prediction, '
        'not evidence of synergy.  To assess synergy, compare this surface against an '
        'experimental checkerboard using the **Synergy Checkerboard** tab.'
    )

    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(
            _combination_heatmap(combo, metric='effect'),
            use_container_width=True, config={'displaylogo': False},
        )
    with h2:
        st.plotly_chart(
            _combination_heatmap(combo, metric='viability'),
            use_container_width=True, config={'displaylogo': False},
        )

    # ── IC50 comparison bar chart ─────────────────────────────────────────────
    ic50_chart = _ic50_summary_bars(pred_a, pred_b, ic50_combo)
    if ic50_chart is not None:
        st.markdown('---')
        st.markdown('#### IC50 Comparison')
        st.caption(
            'IC50 values are **model-predicted** (4PL curves reconstructed from published AUC data).  '
            'The combination IC50 is computed along the equimolar (1:1) dose trajectory under '
            'Bliss independence — not a measured IC50.'
        )
        st.plotly_chart(ic50_chart, use_container_width=True, config={'displaylogo': False})

    # ── Uncertainty ───────────────────────────────────────────────────────────
    with st.expander('Bootstrap uncertainty estimates (perturbation of AUC inputs)', expanded=False):
        st.write(
            'To capture uncertainty from AUC-to-curve reconstruction, we perturb the '
            'source AUC values by ±1 SD (σ ≈ 0.104 from conformal prediction intervals; '
            'σ ≈ 0.15 for Sulfasalazine) and recompute 1,000 bootstrap samples.'
        )
        with st.spinner('Running bootstrap...'):
            try:
                boot = bootstrap_combination_summary(drug_a, drug_b, cell_line,
                                                      n_boot=500, seed=42)
                if 'effect_top_dose' in boot:
                    b = boot['effect_top_dose']
                    st.metric(
                        'Bliss E_AB top-dose (bootstrap mean)',
                        f"{b['mean']:.3f}",
                        help=f"95% CI: [{b['lo95']:.3f}, {b['hi95']:.3f}] over {b['n_boot']} samples"
                    )
                    st.caption(f"95% bootstrap CI: **{b['lo95']:.3f} – {b['hi95']:.3f}**")

                if boot.get('ic50_equimolar_um', {}).get('estimable', True) and \
                   'mean' in boot.get('ic50_equimolar_um', {}):
                    b2 = boot['ic50_equimolar_um']
                    st.metric(
                        'Equimolar combo IC50 (bootstrap mean)',
                        f"{b2['mean']:.3g} µM",
                        help=f"95% CI: [{b2['lo95']:.3g}, {b2['hi95']:.3g}] µM"
                    )
                    st.caption(f"95% bootstrap CI: **{b2['lo95']:.3g} – {b2['hi95']:.3g} µM**")
                else:
                    st.info('Equimolar IC50 not estimable in the majority of bootstrap samples '
                            '(drug does not reach 50% inhibition in this concentration range).')
            except Exception as e:
                st.warning(f'Bootstrap failed: {e}')

    # ── Data sources and biological context ────────────────────────────────────
    st.markdown('---')
    st.markdown('#### Biological Context & Data Sources')

    with st.expander('Biological rationale', expanded=True):
        st.write(combo_cfg['biological_rationale'])
        st.caption(combo_cfg['data_quality'])

    st.markdown('**Data sources for this prediction:**')
    src_rows = [
        {'Drug': drug_a, 'AUC value': pred_a['auc_norm_ctrp'],
         'Cell line': cell_line, 'Source': pred_a['data_source'],
         'Confidence': pred_a['confidence']},
        {'Drug': drug_b, 'AUC value': pred_b['auc_norm_ctrp'],
         'Cell line': cell_line, 'Source': pred_b['data_source'],
         'Confidence': pred_b['confidence']},
    ]
    st.dataframe(pd.DataFrame(src_rows), hide_index=True, use_container_width=True)

    # ── All available AUC data ────────────────────────────────────────────────
    with st.expander('All available AUC data for these drugs across TNBC lines'):
        all_rows = []
        for drug in [drug_a, drug_b]:
            for line, auc in CTRP_AUC.get(drug, {}).items():
                all_rows.append({
                    'Drug': drug, 'Cell line': line,
                    'AUC (norm, 0-1)': round(auc, 4),
                    'Effect proxy (1-AUC)': round(1 - auc, 4),
                    'Source': DATA_SOURCE.get(drug, '—'),
                    'Confidence': CONFIDENCE.get(drug, '—'),
                })
        st.dataframe(pd.DataFrame(all_rows), hide_index=True, use_container_width=True)

    # ── Scientific limitations ─────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('#### ⚠ Scientific Limitations')
    for i, lim in enumerate(combo['limitations'], 1):
        st.markdown(f'**{i}.** {lim}')

    # ── Method summary ─────────────────────────────────────────────────────────
    with st.expander('Methodology: how these predictions are computed'):
        st.markdown("""
**Step 1 — Source AUC values**

For RSL3 and Erastin, we use measured AUCs from CTRPv2.1 (Nat Chem Biol 2016/2017):
area under the sigmoid-fit concentration-response curve on a 0-1 viability scale over
~20 log₁₀ dose units (auc_norm_0_1 = auc_raw / 20; lower = more cell death).

For Sulfasalazine, we use the PRISM-graded model prediction from the FerroSelect sealed panel
(only 9 training rows — LOW confidence).

**Step 2 — 4PL reconstruction**

We parameterise a four-parameter logistic (4PL) curve for each drug:

$$E(d) = E_{\\min} + \\frac{E_{\\max} - E_{\\min}}{1 + 10^{(\\log_{10}\\text{EC}_{50} - \\log_{10}d) \\cdot h}}$$

with:
- $E_{\\min} = 0$ (no effect at zero dose)
- $E_{\\max} \\approx 2(1 - \\text{AUC}_{\\text{norm}})$, clipped to [0.05, 0.98]
- EC50 and Hill slope from literature-informed priors (RSL3: 0.5 µM; Erastin: 3 µM; Sulfasalazine: 300 µM)

**Step 3 — Bliss Independence combination surface**

$$V_{AB}(d_A, d_B) = V_A(d_A) \\times V_B(d_B)$$
$$E_{AB}(d_A, d_B) = E_A + E_B - E_A \\cdot E_B$$

**Step 4 — IC50 (equimolar, 1:1 ratio)**

Along the ray $d_A = d_B = t$, we find $t^*$ such that $V_{AB}(t^*, t^*) = 0.5$ using the Brent method.

**Step 5 — AUC**

AUC is integrated viability vs log₁₀(concentration µM) using the trapezoidal rule.
Lower AUC = greater sensitivity (consistent with CTRPv2.1 convention).

**References**

- Bliss CI (1939). The toxicity of poisons applied jointly. *Ann Appl Biol* 26:585.
- Seashore-Ludlow B et al. (2015). *Nat Chem Biol* 11:883.  
- Viswanathan VS et al. (2017). *Nat Chem Biol* 13:1178.
- Yadav B et al. (2015). *Sci Rep* 5:11705 (ZIP synergy model).
        """)


def render_single_drug_prediction_tab(st):
    """Render the single-drug IC50/AUC prediction tab."""
    if not _MODULE_OK:
        st.error(f'Combination predictor module failed: {_MODULE_ERR}')
        return

    st.subheader('Single-Drug Response Prediction (IC50 & AUC)')
    st.info(
        'Predictions are reconstructed from published/model AUC values using 4PL curve fitting.  '
        'For RSL3 and Erastin, source AUCs come from CTRPv2.1 (measured).  '
        'For Sulfasalazine, the AUC comes from the PRISM-graded FerroSelect model (LOW confidence).'
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        drug = st.selectbox('Drug', ['RSL3', 'ERASTIN', 'SULFASALAZINE'], key='single_drug_sel')
    with col2:
        cell_line = st.selectbox(
            'Cell line', list(SUPPORTED_CELL_LINES.keys()),
            format_func=lambda x: f'{x} ({SUPPORTED_CELL_LINES[x]["subtype"]})',
            key='single_line_sel',
        )
    with col3:
        conc_max = st.number_input('Max concentration (µM)', value=float(
            {'RSL3': 10.0, 'ERASTIN': 100.0, 'SULFASALAZINE': 2000.0}.get(drug, 100.0)
        ), min_value=0.001, format='%g', key='single_conc_max')

    conc_min = conc_max / 10000.0
    try:
        pred = predict_single_drug(drug, cell_line, conc_range=(conc_min, conc_max), n_points=200)
    except ValueError as e:
        st.error(f'Prediction error: {e}')
        return

    # Metric cards
    m1, m2, m3 = st.columns(3)
    m1.metric(
        f'{drug} IC50 ({cell_line})',
        pred['ic50_display'],
        help=f'4PL-predicted IC50 from AUC = {pred["auc_norm_ctrp"]:.4f}. {pred["confidence"]}',
    )
    m2.metric(
        'AUC (source, 0-1 scale)',
        f'{pred["auc_norm_ctrp"]:.4f}',
        help=f'{pred["data_source"]}. Lower = more sensitive.',
    )
    m3.metric(
        'Effect proxy (1 − AUC)',
        f'{1 - pred["auc_norm_ctrp"]:.4f}',
        help='Approximate maximum killing effect extrapolated from the AUC.',
    )

    # Dose-response chart
    color = DRUG_A_COLOR if drug == 'RSL3' else _drug_b_color(drug)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred['conc_um'], y=pred['viability'],
        mode='lines', name=f'{drug} viability',
        line=dict(color=color, width=2.5),
        hovertemplate='Conc: %{x:.3g} µM<br>Viability: %{y:.3f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=pred['conc_um'], y=pred['effect'],
        mode='lines', name=f'{drug} effect',
        line=dict(color=color, width=2.5, dash='dash'),
        hovertemplate='Conc: %{x:.3g} µM<br>Effect: %{y:.3f}<extra></extra>',
    ))
    fig.add_hline(y=0.5, line=dict(dash='dot', color='#868E96', width=1),
                  annotation_text='50% threshold (IC50 / IC50 reference)',
                  annotation_position='bottom right',
                  annotation_font=dict(color='#868E96', size=10))
    if pred['ic50_um'] is not None:
        fig.add_vline(x=pred['ic50_um'], line=dict(dash='dot', color=color, width=1),
                      annotation_text=f'IC50 = {pred["ic50_um"]:.3g} µM',
                      annotation_font=dict(color=color, size=10))
    fig.update_layout(
        title=dict(text=f'{drug} dose-response in {cell_line} (4PL reconstruction from AUC)', font=dict(size=13)),
        xaxis=dict(type='log', title='Concentration (µM)', tickformat='.2g'),
        yaxis=dict(title='Response', range=[-0.05, 1.05]),
        legend=dict(x=0.01, y=0.05, bgcolor='rgba(0,0,0,0)'),
        height=400, margin=dict(t=44, b=40, l=10, r=10),
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False})

    # All lines comparison
    with st.expander('All cell lines for this drug'):
        rows = []
        for line, auc in CTRP_AUC.get(drug, {}).items():
            try:
                p = predict_single_drug(drug, line, conc_range=(conc_min, conc_max))
                rows.append({
                    'Cell line': line,
                    'Subtype': SUPPORTED_CELL_LINES.get(line, {}).get('subtype', '—'),
                    'IC50 (µM)': f'{p["ic50_um"]:.3g}' if p['ic50_um'] is not None else 'Not estimable',
                    'AUC (norm)': f'{auc:.4f}',
                    'Confidence': CONFIDENCE.get(drug, '—'),
                })
            except Exception:
                pass
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.caption(
        f'**Data source:** {pred["data_source"]}  \n'
        f'**Confidence:** {pred["confidence"]}  \n'
        'IC50 and dose-response curve are **model reconstructions**, not direct measurements.  '
        'The 4PL curve is consistent with the published AUC but does not reproduce the exact '
        'experimental dose-response (which is not available in the supplied dataset).'
    )
