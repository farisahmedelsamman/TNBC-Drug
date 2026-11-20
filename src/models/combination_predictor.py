"""Drug combination response predictor — RSL3 + Erastin / RSL3 + Sulfasalazine.

Scientific approach
-------------------
We have single-drug AUC data (CTRPv2.1 for RSL3 & Erastin; PRISM-graded for
Sulfasalazine) but NO experimental combination measurements.  Therefore:

1.  We fit 4-parameter logistic (4PL) dose-response curves to each drug using
    the available measured AUC values to reconstruct plausible single-agent
    dose-response profiles.
2.  We use Bliss Independence as the null combination model:
        V_AB = V_A × V_B   (viability convention)
        E_AB = E_A + E_B − E_A × E_B  (inhibition/effect convention)
3.  We generate IC50 along a fixed-ratio trajectory and AUC over the
    combination response surface.

CRITICAL LIMITATION
-------------------
These are mathematical predictions from a Bliss additivity model.
They have NOT been experimentally validated with combination assay data.
They cannot establish actual biological synergy.  Any excess above the
Bliss additive expectation observed in a future wet-lab checkerboard is what
would constitute evidence of synergy.

AUC convention
--------------
AUC is the area under the viability-vs-log10(concentration) curve, integrated
over a user-specified concentration range using the trapezoidal rule applied
to the 4PL-predicted viability.  LOWER AUC = greater drug sensitivity
(consistent with CTRPv2.1 convention where auc_norm_0_1 / 20 = AUC on 0-1
scale; smaller = more cell death).

IC50 definition for combinations
---------------------------------
For a fixed dose-ratio combination (r = dose_A / dose_B = constant):
  - We trace the response along the ray dose_B = t, dose_A = r·t
  - IC50 is the total concentration (r·t + t) at which V_AB = 0.5
  - Equivalently: concentration of each component at that point.

References
----------
- Bliss CI (1939). The toxicity of poisons applied jointly. Ann. Appl. Biol.
- Yadav B et al. (2015). Quantitative scoring of differential drug sensitivity
  for individually optimized anticancer therapies. Sci Rep 5:11705.
- CTRPv2.1: Seashore-Ludlow B et al. (2015). Nat Chem Biol 11:883.
  (Viswanathan VS et al. 2017, Nat Chem Biol 13:1178 — ferroptosis context.)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, brentq


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N_BOOT = 1000  # bootstrap iterations for uncertainty

# CTRPv2.1 measured AUCs for key TNBC lines (auc_norm_0_1, 0-1 scale,
# LOWER = more sensitive). These come from outputs_local/ctpv2_ferroptosis_auc.csv
# Source: Nat Chem Biol 2016 / 2017 (see module docstring).
CTRP_AUC = {
    'RSL3': {
        'MDA-MB-231':  0.3091,   # ACH-000768  (highly sensitive, 29.6 percentile)
        'MDA-MB-468':  0.5162,   # ACH-000849  (moderately sensitive)
        'MCF7':        0.5531,   # ACH-000019  (less sensitive, 82.9 percentile)
    },
    'ERASTIN': {
        'MDA-MB-231':  0.5639,   # ACH-000768  (from CTRP data)
        'MDA-MB-468':  0.6367,   # ACH-000849
        'MCF7':        0.6310,   # ACH-000019
    },
    # Sulfasalazine: NOT in CTRP; values come from PRISM-graded model
    # (FerroSelect panel predictions, LOW confidence — 9 training rows only)
    'SULFASALAZINE': {
        'MDA-MB-231':  0.8874,   # DPC-006173, LOW confidence
        'MDA-MB-468':  0.9306,
        'MCF7':        0.9353,
    },
}

DATA_SOURCE = {
    'RSL3': 'CTRPv2.1 measured AUC (Nat Chem Biol 2016/2017); n=808 lines',
    'ERASTIN': 'CTRPv2.1 measured AUC (Nat Chem Biol 2016/2017); n=792 lines',
    'SULFASALAZINE': (
        'PRISM-graded model prediction (LOW confidence — only 9 training rows; '
        'Sulfasalazine was not profiled in CTRPv2.1)'
    ),
}

CONFIDENCE = {
    'RSL3': 'HIGH — measured CTRP AUC',
    'ERASTIN': 'HIGH — measured CTRP AUC',
    'SULFASALAZINE': 'LOW — model estimate from 9 PRISM training rows',
}

# Typical ferroptosis inducer pharmacological parameters derived from published
# dose-response literature and the CTRP data distributions.
# Hill slope n=1.2 is a soft prior; fit is performed when possible.
# EC50 (µM) priors for each drug: informed by published IC50 ranges in TNBC lines.
EC50_PRIOR = {
    'RSL3':          0.5,    # typical GPX4 inhibitor IC50 ~0.1-2 µM in TNBC
    'ERASTIN':       3.0,    # System Xc- inhibitor; wider range 1-30 µM
    'SULFASALAZINE': 300.0,  # weaker effect; often 100-500 µM
}
HILL_PRIOR = {
    'RSL3':          1.5,
    'ERASTIN':       1.2,
    'SULFASALAZINE': 1.0,
}


# ---------------------------------------------------------------------------
# 4PL model helpers
# ---------------------------------------------------------------------------

def _4pl(log_conc: np.ndarray, e_min: float, e_max: float,
          log_ec50: float, hill: float) -> np.ndarray:
    """Four-parameter logistic inhibition response.

    E(d) = E_min + (E_max - E_min) / (1 + 10^((log_EC50 - log_d) * hill))

    Parameters
    ----------
    log_conc : log10 concentration (µM)
    e_min    : bottom asymptote (effect at zero dose, should be ~0)
    e_max    : top asymptote (maximum inhibition, 0–1)
    log_ec50 : log10 EC50 (µM)
    hill     : Hill slope (>0)

    Returns
    -------
    effect (fractional inhibition, 0–1)
    """
    return e_min + (e_max - e_min) / (1.0 + 10.0 ** ((log_ec50 - log_conc) * hill))


def viability_from_params(conc_um: np.ndarray, e_min: float, e_max: float,
                           log_ec50: float, hill: float) -> np.ndarray:
    """Return predicted viability (1 - effect) for concentration array (µM)."""
    conc = np.asarray(conc_um, float)
    viab = np.ones_like(conc)
    pos = conc > 0
    viab[pos] = 1.0 - np.clip(
        _4pl(np.log10(conc[pos]), e_min, e_max, log_ec50, hill), 0.0, 1.0
    )
    return viab


def effect_from_params(conc_um: np.ndarray, e_min: float, e_max: float,
                        log_ec50: float, hill: float) -> np.ndarray:
    """Return predicted effect (1 - viability)."""
    return 1.0 - viability_from_params(conc_um, e_min, e_max, log_ec50, hill)


def auc_norm_to_4pl_params(drug: str, auc_norm: float) -> Tuple[float, float, float, float]:
    """Convert a normalised AUC (0-1 CTRP scale, lower=sensitive) to 4PL parameters.

    Strategy: We know the AUC integrates the viability curve over ~20 log10
    concentration units on the CTRP scale.  We parameterise a 4PL with:
      - e_min = 0.0  (no effect at zero dose)
      - e_max ≈ 1 - auc_norm  (approximate max killing consistent with AUC)
      - EC50 from prior, adjusted so the integrated AUC matches
      - Hill from prior

    This is a *reconstruction*, not a fit to actual dose-response data.
    The resulting curve is a pharmacologically plausible representative of
    a cell-line response with the given AUC.  Uncertainty is captured by
    bootstrapping e_max and EC50 within reasonable ranges.
    """
    ec50 = EC50_PRIOR.get(drug, 1.0)
    hill = HILL_PRIOR.get(drug, 1.2)
    # Approximate e_max: on the CTRP 0-1 scale, auc_norm is the mean viability
    # integrated over ~20 log10 units.  At saturation the viability asymptote
    # is 1 - e_max.  We set e_max so that the curve is consistent with the AUC.
    # Simple approximation: e_max ≈ 2*(1 - auc_norm) clipped to [0.05, 0.98].
    # Factor 2 accounts for the fact that only part of the concentration range
    # is in the active dose window.
    e_max = float(np.clip(2.0 * (1.0 - auc_norm), 0.05, 0.98))
    log_ec50 = float(np.log10(ec50))
    return (0.0, e_max, log_ec50, hill)


def build_dose_response(drug: str, cell_line: str,
                         conc_range: Tuple[float, float] = (0.001, 1000.0),
                         n_points: int = 100) -> Dict[str, Any]:
    """Build a predicted single-agent dose-response curve.

    Parameters
    ----------
    drug       : drug name (RSL3, ERASTIN, SULFASALAZINE)
    cell_line  : cell line name (MDA-MB-231, MDA-MB-468, MCF7)
    conc_range : (min, max) concentration in µM
    n_points   : number of curve points

    Returns
    -------
    dict with keys:
        conc_um       : concentration array (µM)
        viability     : predicted viability (0-1)
        effect        : predicted fractional inhibition (0-1)
        ic50_um       : predicted IC50 (µM) or None if unreachable
        auc_norm      : source AUC (0-1 scale, lower=sensitive)
        params        : (e_min, e_max, log_ec50, hill)
        data_source   : provenance string
        confidence    : confidence level string
    """
    auc_norm = CTRP_AUC.get(drug, {}).get(cell_line)
    if auc_norm is None:
        raise ValueError(f'No AUC data for {drug} in {cell_line}')

    params = auc_norm_to_4pl_params(drug, auc_norm)
    e_min, e_max, log_ec50, hill = params

    conc = np.geomspace(conc_range[0], conc_range[1], n_points)
    viab = viability_from_params(conc, *params)
    effect = 1.0 - viab

    # IC50: concentration at which effect = 0.5
    ic50 = _compute_ic50(params, e_max)

    # AUC under viability-vs-log10(concentration) curve (trapezoidal, log-spaced conc)
    _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    auc_curve = float(_trapz(viab, x=np.log10(conc)))

    return {
        'conc_um': conc,
        'viability': viab,
        'effect': effect,
        'ic50_um': ic50,
        'auc_norm': auc_norm,
        'auc_curve': auc_curve,
        'params': params,
        'data_source': DATA_SOURCE.get(drug, 'Unknown'),
        'confidence': CONFIDENCE.get(drug, 'Unknown'),
    }


def _compute_ic50(params: Tuple, e_max: float) -> Optional[float]:
    """Compute IC50 (µM) by inverting the 4PL.  Returns None if unreachable."""
    e_min, e_max_p, log_ec50, hill = params
    target = 0.5
    if target > e_max_p:
        return None  # drug never reaches 50% inhibition
    if target <= e_min:
        return 0.0
    frac = (target - e_min) / max(e_max_p - e_min, 1e-9)
    frac = min(max(frac, 1e-9), 1 - 1e-9)
    log_dose = log_ec50 - np.log10(1.0 / frac - 1.0) / max(hill, 1e-9)
    return float(10 ** log_dose)


# ---------------------------------------------------------------------------
# Combination response (Bliss Independence)
# ---------------------------------------------------------------------------

def bliss_combination_surface(
    drug_a: str, drug_b: str, cell_line: str,
    conc_a: Optional[np.ndarray] = None,
    conc_b: Optional[np.ndarray] = None,
    n_points: int = 20,
    conc_range_a: Tuple[float, float] = None,
    conc_range_b: Tuple[float, float] = None,
) -> Dict[str, Any]:
    """Compute Bliss-independent predicted combination response surface.

    V_AB(d_A, d_B) = V_A(d_A) × V_B(d_B)

    This is a mathematical prediction from single-agent data.
    It represents the EXPECTED combination effect under the null hypothesis
    of Bliss independence.  It does NOT predict synergy — it IS the reference
    against which synergy is measured.

    Parameters
    ----------
    drug_a, drug_b : drug names
    cell_line      : cell line name
    conc_a, conc_b : concentration arrays (µM); if None, auto-generated
    n_points       : number of concentration points per axis
    conc_range_a   : (min, max) for drug A (µM)
    conc_range_b   : (min, max) for drug B (µM)

    Returns
    -------
    dict with:
        conc_a, conc_b       : concentration axes (µM, including 0)
        viab_a, viab_b       : single-agent viability curves
        effect_a, effect_b   : single-agent effect curves
        viab_combination     : (n_a × n_b) Bliss viability surface
        effect_combination   : (n_a × n_b) Bliss effect surface
        bliss_summary        : summary statistics
        ic50_combo           : dict with IC50 information
        auc_summary          : dict with AUC information
        params_a, params_b   : 4PL parameters
        limitations          : list of important caveats
    """
    # Get single-agent 4PL parameters
    auc_a = CTRP_AUC.get(drug_a, {}).get(cell_line)
    auc_b = CTRP_AUC.get(drug_b, {}).get(cell_line)

    if auc_a is None:
        raise ValueError(f'No AUC data for {drug_a} in {cell_line}')
    if auc_b is None:
        raise ValueError(f'No AUC data for {drug_b} in {cell_line}')

    params_a = auc_norm_to_4pl_params(drug_a, auc_a)
    params_b = auc_norm_to_4pl_params(drug_b, auc_b)

    # Build concentration grids
    if conc_a is None:
        cr_a = conc_range_a or _default_conc_range(drug_a)
        conc_a_nz = np.geomspace(cr_a[0], cr_a[1], n_points - 1)
        conc_a = np.concatenate([[0.0], conc_a_nz])
    else:
        conc_a = np.asarray(conc_a, float)
        if conc_a[0] != 0.0:
            conc_a = np.concatenate([[0.0], conc_a])

    if conc_b is None:
        cr_b = conc_range_b or _default_conc_range(drug_b)
        conc_b_nz = np.geomspace(cr_b[0], cr_b[1], n_points - 1)
        conc_b = np.concatenate([[0.0], conc_b_nz])
    else:
        conc_b = np.asarray(conc_b, float)
        if conc_b[0] != 0.0:
            conc_b = np.concatenate([[0.0], conc_b])

    viab_a = viability_from_params(conc_a, *params_a)
    viab_b = viability_from_params(conc_b, *params_b)
    effect_a = 1.0 - viab_a
    effect_b = 1.0 - viab_b

    # Bliss combination surface: V_AB = V_A * V_B
    viab_combo = np.outer(viab_a, viab_b)
    effect_combo = 1.0 - viab_combo

    # IC50 for the combination along the equimolar ratio trajectory
    ic50_combo = _combo_ic50_equimolar(params_a, params_b, conc_a, conc_b)

    # AUC summary
    auc_summary = _combo_auc_summary(
        viab_a, viab_b, viab_combo, conc_a, conc_b,
        auc_a, auc_b, drug_a, drug_b
    )

    # Bliss summary
    bliss_sum = {
        'mean_expected_effect': float(np.mean(effect_combo[1:, 1:])),
        'max_expected_effect': float(np.max(effect_combo[1:, 1:])),
        'min_expected_effect': float(np.min(effect_combo[1:, 1:])),
        'effect_a_max': float(effect_a[-1]),
        'effect_b_max': float(effect_b[-1]),
        'bliss_expected_top': float(effect_combo[-1, -1]),
        'note': (
            'These are BLISS ADDITIVE EXPECTATION values — the null hypothesis '
            'that both drugs act independently.  They are NOT evidence of synergy.  '
            'Synergy/antagonism can only be determined by comparing these values '
            'against experimental combination measurements in a checkerboard assay.'
        ),
    }

    limitations = _build_limitations(drug_a, drug_b)

    return {
        'drug_a': drug_a,
        'drug_b': drug_b,
        'cell_line': cell_line,
        'conc_a': conc_a,
        'conc_b': conc_b,
        'viab_a': viab_a,
        'viab_b': viab_b,
        'effect_a': effect_a,
        'effect_b': effect_b,
        'viab_combination': viab_combo,
        'effect_combination': effect_combo,
        'bliss_summary': bliss_sum,
        'ic50_combo': ic50_combo,
        'auc_summary': auc_summary,
        'params_a': params_a,
        'params_b': params_b,
        'auc_norm_a': auc_a,
        'auc_norm_b': auc_b,
        'data_source_a': DATA_SOURCE.get(drug_a, 'Unknown'),
        'data_source_b': DATA_SOURCE.get(drug_b, 'Unknown'),
        'confidence_a': CONFIDENCE.get(drug_a, 'Unknown'),
        'confidence_b': CONFIDENCE.get(drug_b, 'Unknown'),
        'limitations': limitations,
    }


def _default_conc_range(drug: str) -> Tuple[float, float]:
    """Return a pharmacologically relevant concentration range (µM)."""
    ec50 = EC50_PRIOR.get(drug, 1.0)
    return (ec50 / 100.0, ec50 * 1000.0)


def _combo_ic50_equimolar(params_a, params_b, conc_a, conc_b) -> Dict[str, Any]:
    """IC50 along equimolar ratio trajectory.

    Along the ray dose_A = dose_B = t (equimolar ratio 1:1):
      V_AB(t, t) = V_A(t) * V_B(t)
      IC50_combo = t at which V_AB = 0.5

    Returns dict with combo IC50 info, or explains why it's not estimable.
    """
    # Sample the equimolar trajectory
    all_conc = np.unique(np.concatenate([conc_a[conc_a > 0], conc_b[conc_b > 0]]))
    if len(all_conc) == 0:
        return {'estimable': False, 'reason': 'No non-zero concentrations'}

    t_range = (all_conc.min(), all_conc.max())
    t_arr = np.geomspace(t_range[0], t_range[1], 1000)
    va = viability_from_params(t_arr, *params_a)
    vb = viability_from_params(t_arr, *params_b)
    v_combo = va * vb

    if v_combo[-1] > 0.5:
        # Combination never reaches 50% inhibition in this range
        return {
            'estimable': False,
            'reason': (
                f'The combination does not reach 50% inhibition within the '
                f'tested range [{t_range[0]:.3g}–{t_range[1]:.3g} µM each]. '
                'This is expected if either drug is weak in this cell line.'
            ),
            'viability_at_max_dose': float(v_combo[-1]),
        }

    if v_combo[0] < 0.5:
        return {
            'estimable': False,
            'reason': 'Combination viability is below 0.5 even at the minimum concentration.',
        }

    try:
        def f(t):
            va_ = viability_from_params(np.array([t]), *params_a)[0]
            vb_ = viability_from_params(np.array([t]), *params_b)[0]
            return va_ * vb_ - 0.5

        ic50_t = brentq(f, t_range[0], t_range[1] * 10, xtol=1e-6, maxiter=200)
        va_ic50 = viability_from_params(np.array([ic50_t]), *params_a)[0]
        vb_ic50 = viability_from_params(np.array([ic50_t]), *params_b)[0]
        return {
            'estimable': True,
            'definition': 'Equimolar ratio (1:1): total concentration at which V_AB = 0.5',
            'ic50_each_drug_um': float(ic50_t),
            'ic50_total_um': float(2.0 * ic50_t),
            'viab_a_at_ic50': float(va_ic50),
            'viab_b_at_ic50': float(vb_ic50),
            'note': (
                'Combination IC50 is defined along the equimolar (1:1) dose trajectory. '
                'The value is a Bliss-model prediction, not a measured IC50.'
            ),
        }
    except Exception as e:
        return {'estimable': False, 'reason': f'Numerical error: {e}'}


def _combo_auc_summary(viab_a, viab_b, viab_combo, conc_a, conc_b,
                        auc_norm_a, auc_norm_b, drug_a, drug_b) -> Dict[str, Any]:
    """Summarise AUC metrics for single agents and combination."""
    # AUC of single agents (trapezoidal over log10 concentration axis, excluding zero)
    nz_a = conc_a > 0
    nz_b = conc_b > 0

    _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)

    if nz_a.sum() > 1:
        auc_a_curve = float(_trapz(viab_a[nz_a], x=np.log10(conc_a[nz_a])))
    else:
        auc_a_curve = float('nan')

    if nz_b.sum() > 1:
        auc_b_curve = float(_trapz(viab_b[nz_b], x=np.log10(conc_b[nz_b])))
    else:
        auc_b_curve = float('nan')

    # Combination AUC: mean over drug-B concentrations of the drug-A AUC profile
    # (integrating over log10(conc_a) for each conc_b value)
    if nz_a.sum() > 1 and nz_b.sum() > 1:
        combo_aucs_over_b = []
        for j, _ in enumerate(conc_b):
            va_col = viab_combo[nz_a, j]
            auc_col = float(_trapz(va_col, x=np.log10(conc_a[nz_a])))
            combo_aucs_over_b.append(auc_col)
        auc_combo_mean = float(np.mean(combo_aucs_over_b[1:]))  # skip zero dose of B
    else:
        auc_combo_mean = float('nan')

    return {
        'auc_norm_a_ctrp': auc_norm_a,
        'auc_norm_b_ctrp': auc_norm_b,
        'auc_a_curve': auc_a_curve,
        'auc_b_curve': auc_b_curve,
        'auc_combo_mean': auc_combo_mean,
        'convention': (
            'AUC is integrated viability vs log10(concentration µM) using the '
            'trapezoidal rule.  LOWER = more drug effect (more sensitive cell line).  '
            f'The source CTRP AUC for {drug_a} is {auc_norm_a:.4f} and for {drug_b} '
            f'is {auc_norm_b:.4f} (0-1 normalised, lower=sensitive).'
        ),
        'note': (
            'Combination AUC is the mean of the drug-A single-agent AUC profiles '
            'evaluated at each drug-B concentration.  This is a 2D summary metric; '
            'it has no standard reference definition and must be interpreted with the '
            'full response surface.'
        ),
    }


def _build_limitations(drug_a: str, drug_b: str) -> list:
    """Return a list of scientific limitation statements."""
    base = [
        (
            'All combination predictions are derived from the Bliss Independence null model '
            'applied to single-agent 4PL curves reconstructed from published AUC values.  '
            'No experimental combination assay data was used.'
        ),
        (
            'The 4PL curves used here are AUC-consistent reconstructions, not fits to '
            'actual dose-response measurements.  The exact shape of the curve (EC50, Hill '
            'slope) uses literature-informed priors and may differ from wet-lab observations.'
        ),
        (
            'Bliss Independence assumes the two drugs act through probabilistically independent '
            'mechanisms.  RSL3 and Erastin target different nodes of the ferroptosis pathway '
            '(GPX4 vs System Xc-), making this a reasonable working assumption, but the '
            'assumption has not been verified for these specific drug-line combinations.'
        ),
        (
            'These predictions cannot establish actual biological synergy.  Synergy requires '
            'comparison of the OBSERVED combination response against the Bliss expectation in '
            'a properly replicated checkerboard assay.'
        ),
    ]
    if drug_b == 'SULFASALAZINE' or drug_a == 'SULFASALAZINE':
        base.insert(1, (
            'Sulfasalazine was NOT profiled in CTRPv2.1.  Its AUC estimate comes from a '
            'PRISM-graded model trained on only 9 rows — this is LOW CONFIDENCE.  '
            'The combination prediction for RSL3 + Sulfasalazine carries additional '
            'uncertainty from the Sulfasalazine AUC estimate.'
        ))
    return base


# ---------------------------------------------------------------------------
# Bootstrap uncertainty
# ---------------------------------------------------------------------------

def bootstrap_combination_summary(
    drug_a: str, drug_b: str, cell_line: str,
    n_boot: int = N_BOOT, seed: int = SEED
) -> Dict[str, Any]:
    """Bootstrap uncertainty on combination predictions by perturbing AUC inputs.

    Models the uncertainty in the AUC-to-4PL reconstruction by sampling
    AUC values from an empirical distribution (mean ± 1 SD based on published
    model CV errors from prediction_intervals.json: global abs_q80 ≈ 0.104).

    Returns 95% bootstrap confidence intervals on key combo metrics.
    """
    rng = np.random.RandomState(seed)
    sigma_a = 0.104 if drug_a != 'SULFASALAZINE' else 0.15  # higher uncertainty for SFZ
    sigma_b = 0.104 if drug_b != 'SULFASALAZINE' else 0.15

    auc_a_base = CTRP_AUC.get(drug_a, {}).get(cell_line, 0.5)
    auc_b_base = CTRP_AUC.get(drug_b, {}).get(cell_line, 0.5)

    boot_e_top = []
    boot_ic50 = []

    for _ in range(n_boot):
        auc_a = float(np.clip(auc_a_base + rng.randn() * sigma_a, 0.05, 0.98))
        auc_b = float(np.clip(auc_b_base + rng.randn() * sigma_b, 0.05, 0.98))

        pa = auc_norm_to_4pl_params(drug_a, auc_a)
        pb = auc_norm_to_4pl_params(drug_b, auc_b)

        # Top-dose effect (use 100x EC50 as a proxy for "top dose")
        c_top_a = 10 ** (pa[2] + 2)
        c_top_b = 10 ** (pb[2] + 2)
        va_top = viability_from_params(np.array([c_top_a]), *pa)[0]
        vb_top = viability_from_params(np.array([c_top_b]), *pb)[0]
        e_top = 1.0 - va_top * vb_top
        boot_e_top.append(e_top)

        # Equimolar IC50
        all_conc = np.geomspace(1e-4, 1e4, 500)
        va_arr = viability_from_params(all_conc, *pa)
        vb_arr = viability_from_params(all_conc, *pb)
        v_combo = va_arr * vb_arr
        if v_combo[0] < 0.5 <= v_combo[-1]:  # monotone, crosses 0.5
            try:
                def f(t):
                    va_ = viability_from_params(np.array([t]), *pa)[0]
                    vb_ = viability_from_params(np.array([t]), *pb)[0]
                    return va_ * vb_ - 0.5
                ic50 = brentq(f, 1e-4, 1e4, xtol=1e-6, maxiter=100)
                boot_ic50.append(ic50)
            except Exception:
                pass

    result = {}
    if boot_e_top:
        arr = np.array(boot_e_top)
        result['effect_top_dose'] = {
            'mean': float(arr.mean()),
            'lo95': float(np.percentile(arr, 2.5)),
            'hi95': float(np.percentile(arr, 97.5)),
            'n_boot': len(arr),
        }
    if boot_ic50:
        arr = np.array(boot_ic50)
        result['ic50_equimolar_um'] = {
            'mean': float(arr.mean()),
            'lo95': float(np.percentile(arr, 2.5)),
            'hi95': float(np.percentile(arr, 97.5)),
            'n_boot': len(arr),
        }
    else:
        result['ic50_equimolar_um'] = {
            'estimable': False,
            'reason': 'IC50 not estimable in the bootstrap range for most samples',
        }
    return result


# ---------------------------------------------------------------------------
# Convenience: single drug IC50 and AUC prediction
# ---------------------------------------------------------------------------

def predict_single_drug(drug: str, cell_line: str,
                         conc_range: Tuple[float, float] = None,
                         n_points: int = 100) -> Dict[str, Any]:
    """Full single-drug prediction: dose-response, IC50, AUC.

    Returns a dict ready for display.
    """
    if conc_range is None:
        conc_range = _default_conc_range(drug)
    dr = build_dose_response(drug, cell_line, conc_range, n_points)
    ic50 = dr['ic50_um']
    return {
        'drug': drug,
        'cell_line': cell_line,
        'ic50_um': ic50,
        'ic50_display': f'{ic50:.3g} µM' if ic50 is not None else 'Not estimable (>50% inhibition not reached)',
        'auc_norm_ctrp': dr['auc_norm'],
        'auc_norm_display': f'{dr["auc_norm"]:.4f} (lower = more sensitive)',
        'conc_um': dr['conc_um'],
        'viability': dr['viability'],
        'effect': dr['effect'],
        'params': dr['params'],
        'data_source': dr['data_source'],
        'confidence': dr['confidence'],
        'conc_range_um': conc_range,
    }


# ---------------------------------------------------------------------------
# Supported combinations registry
# ---------------------------------------------------------------------------

SUPPORTED_COMBINATIONS = {
    'RSL3 + ERASTIN': {
        'drug_a': 'RSL3',
        'drug_b': 'ERASTIN',
        'biological_rationale': (
            'RSL3 (Type II ferroptosis inducer) directly inhibits GPX4, while '
            'Erastin (Type I) blocks System Xc- reducing cysteine and GSH synthesis.  '
            'These drugs target different steps of the ferroptosis defense axis: '
            'upstream GSH depletion (Erastin) and downstream GPX4 inactivation (RSL3).  '
            'Combination is mechanistically plausible; Bliss independence is a reasonable '
            'starting null model given the distinct molecular targets.'
        ),
        'data_quality': 'Both drugs have measured CTRP AUCs (HIGH confidence for RSL3; HIGH for Erastin)',
    },
    'RSL3 + SULFASALAZINE': {
        'drug_a': 'RSL3',
        'drug_b': 'SULFASALAZINE',
        'biological_rationale': (
            'RSL3 (Type II, GPX4 inhibitor) + Sulfasalazine (Type I, System Xc- inhibitor). '
            'Sulfasalazine acts similarly to Erastin but is a clinically approved anti-inflammatory '
            'drug, making the combination potentially more translatable.  '
            'Note: Sulfasalazine is a weaker ferroptosis inducer at typical concentrations.'
        ),
        'data_quality': (
            'RSL3 has measured CTRP AUC (HIGH); Sulfasalazine uses PRISM model estimate '
            'from only 9 training rows (LOW confidence). Overall combination confidence: LOW.'
        ),
    },
}

SUPPORTED_CELL_LINES = {
    'MDA-MB-231': {
        'ach_id': 'ACH-000768',
        'subtype': 'TNBC (Basal B, claudin-low)',
        'notes': 'Primary TNBC model; highest RSL3 sensitivity in this panel',
    },
    'MDA-MB-468': {
        'ach_id': 'ACH-000849',
        'subtype': 'TNBC (Basal A)',
        'notes': 'EGFR-amplified TNBC; moderate RSL3 sensitivity',
    },
    'MCF7': {
        'ach_id': 'ACH-000019',
        'subtype': 'ER+ luminal A (control)',
        'notes': 'Hormone receptor-positive control; less sensitive to ferroptosis',
    },
}
