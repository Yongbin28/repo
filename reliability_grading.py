import pandas as pd
import numpy as np

def compute_golden_correlation(candidate_stats: dict, golden_baseline: dict) -> float:
    """
    Compute Pearson correlation between a wafer's per-parameter means and the
    virtual golden baseline means. Returns NaN when there is not enough overlap.
    """
    if not candidate_stats or not golden_baseline or not golden_baseline.get("params"):
        return float("nan")

    candidate_means = []
    golden_means = []

    for param, g_stats in golden_baseline["params"].items():
        c_stats = candidate_stats.get(param)
        if not c_stats:
            continue

        c_mean = pd.to_numeric(c_stats.get("mean"), errors="coerce")
        g_mean = pd.to_numeric(g_stats.get("mean"), errors="coerce")
        if pd.isna(c_mean) or pd.isna(g_mean):
            continue

        candidate_means.append(float(c_mean))
        golden_means.append(float(g_mean))

    if len(candidate_means) < 2:
        return float("nan")

    corr = np.corrcoef(candidate_means, golden_means)[0, 1]
    if not np.isfinite(corr):
        return float("nan")

    return float(np.clip(corr, -1.0, 1.0))


def compute_max_zscore(candidate_stats: dict, golden_baseline: dict) -> float:
    """
    Compute the maximum absolute Z-score of matching parameters.
    """
    if not candidate_stats or not golden_baseline or not golden_baseline.get("params"):
        return 0.0

    max_z = 0.0
    for param, g_stats in golden_baseline["params"].items():
        c_stats = candidate_stats.get(param)
        if not c_stats:
            continue

        c_mean = pd.to_numeric(c_stats.get("mean"), errors="coerce")
        g_mean = pd.to_numeric(g_stats.get("mean"), errors="coerce")
        g_std = pd.to_numeric(g_stats.get("std"), errors="coerce")

        if pd.isna(c_mean) or pd.isna(g_mean) or pd.isna(g_std) or g_std <= 1e-9:
            continue

        max_z = max(max_z, abs(c_mean - g_mean) / g_std)

    return float(max_z)


def compute_zscore_closeness(candidate_stats: dict, golden_baseline: dict) -> float:
    """
    Compute closeness score based on average absolute Z-score of matching parameters.
    Applies a progressive penalty if the maximum absolute Z-score exceeds 2.0 (marginal failure).
    Returns a score between 0.0 and 100.0.
    """
    if not candidate_stats or not golden_baseline or not golden_baseline.get("params"):
        return 100.0

    dev_zs = []
    for param, g_stats in golden_baseline["params"].items():
        c_stats = candidate_stats.get(param)
        if not c_stats:
            continue

        c_mean = pd.to_numeric(c_stats.get("mean"), errors="coerce")
        g_mean = pd.to_numeric(g_stats.get("mean"), errors="coerce")
        g_std = pd.to_numeric(g_stats.get("std"), errors="coerce")

        if pd.isna(c_mean) or pd.isna(g_mean) or pd.isna(g_std) or g_std <= 1e-9:
            continue

        dev_zs.append(abs(c_mean - g_mean) / g_std)

    if not dev_zs:
        return 100.0

    mean_z = float(np.mean(dev_zs))
    # Map average absolute Z-score to closeness percentage:
    # 0 deviation -> 100% closeness.
    # Deviation of >= 4.0 -> 0% closeness.
    closeness = float(np.clip(100.0 - 25.0 * mean_z, 0.0, 100.0))

    # Apply penalty based on maximum absolute Z-score
    max_z = float(np.max(dev_zs))
    if max_z > 1.0:
        penalty = 20.0 + 30.0 * (max_z - 1.0)
        penalty = np.clip(penalty, 0.0, 80.0)
        closeness = max(0.0, closeness - penalty)

    return float(closeness)


def compute_reliability_score(
    wafer_id: str,
    mttf_years: float,
    golden_correlation: float = 0.0,
    predicted_yield: float = 1.0,
    golden_similarity_probe: float = None,
    golden_similarity_ft: float = None,
    probe_corr: float = None,
    probe_z_closeness: float = None,
    ft_corr: float = None,
    ft_z_closeness: float = None,
    probe_max_z: float = None,
    ft_max_z: float = None,
) -> dict:
    """
    Standalone post-packaging reliability score using package-physics MTTF,
    Wafer Probe Golden Similarity, and Final Test Golden Similarity.
    """
    corr_value = pd.to_numeric(golden_correlation, errors="coerce")
    if pd.isna(corr_value):
        corr_value = 0.0

    # Map [-1, 1] correlation to [0, 100], with negative correlation treated as worst-case.
    s_corr = np.clip(max(float(corr_value), 0.0) * 100.0, 0, 100)

    # Compute Z-score penalties if max Z is provided
    probe_penalty = 0.0
    if probe_max_z is not None and probe_max_z > 1.0:
        probe_penalty = np.clip(20.0 + 30.0 * (probe_max_z - 1.0), 0.0, 80.0)

    ft_penalty = 0.0
    if ft_max_z is not None and ft_max_z > 1.0:
        ft_penalty = np.clip(20.0 + 30.0 * (ft_max_z - 1.0), 0.0, 80.0)

    # Adjust the overall correlation score using the maximum of the two penalties
    s_corr = float(np.clip(s_corr - max(probe_penalty, ft_penalty), 0.0, 100.0))

    # Setup defaults for new granular metrics (for backward compatibility)
    if probe_corr is None:
        probe_corr = corr_value
    probe_corr_val = pd.to_numeric(probe_corr, errors="coerce")
    if pd.isna(probe_corr_val):
        probe_corr_val = 0.0
    s_probe_corr = np.clip(max(float(probe_corr_val), 0.0) * 100.0, 0, 100)
    s_probe_corr = float(np.clip(s_probe_corr - probe_penalty, 0.0, 100.0))

    if ft_corr is None:
        ft_corr = corr_value
    ft_corr_val = pd.to_numeric(ft_corr, errors="coerce")
    if pd.isna(ft_corr_val):
        ft_corr_val = 0.0
    s_ft_corr = np.clip(max(float(ft_corr_val), 0.0) * 100.0, 0, 100)
    s_ft_corr = float(np.clip(s_ft_corr - ft_penalty, 0.0, 100.0))

    if probe_z_closeness is None:
        probe_z_closeness = s_probe_corr

    if ft_z_closeness is None:
        ft_z_closeness = s_ft_corr

    if golden_similarity_probe is None:
        golden_similarity_probe = 0.5 * s_probe_corr + 0.5 * probe_z_closeness

    if golden_similarity_ft is None:
        golden_similarity_ft = 0.5 * s_ft_corr + 0.5 * ft_z_closeness

    target_mttf = 15.0
    s_mttf = np.clip((mttf_years / target_mttf) * 100, 0, 100)

    # Updated formula: 30% Probe Golden Similarity + 30% FT Golden Similarity + 40% MTTF Score
    composite_score = (
        (0.30 * golden_similarity_probe) +
        (0.30 * golden_similarity_ft) +
        (0.40 * s_mttf)
    )

    yield_penalty = 0.0
    if predicted_yield < 0.80:
        yield_penalty = np.clip((0.80 - predicted_yield) * 100, 0, 25)

    composite_score = float(np.round(np.clip(composite_score - yield_penalty, 0, 100), 4))

    return {
        "Risk_Score": composite_score,
        "Components": {
            "Golden_Correlation": float(corr_value),
            "Golden_Correlation_Score": float(s_corr),
            "MTTF_Score": float(s_mttf),
            "Yield_Penalty": float(yield_penalty),
            "Probe_Similarity": float(golden_similarity_probe),
            "FT_Similarity": float(golden_similarity_ft),
            "Probe_Correlation": float(probe_corr_val),
            "Probe_Correlation_Score": float(s_probe_corr),
            "Probe_Z_Closeness": float(probe_z_closeness),
            "FT_Correlation": float(ft_corr_val),
            "FT_Correlation_Score": float(s_ft_corr),
            "FT_Z_Closeness": float(ft_z_closeness),
        },
    }


def compute_product_grade(
    risk_score: float,
    mttf_years: float,
    golden_correlation: float,
    predicted_yield: float = 1.0,
) -> dict:
    """
    Standalone post-packaging product grade classification.
    """
    if risk_score >= 90:
        grade = "A"
        desc = "Automotive"
    elif risk_score >= 80:
        grade = "B"
        desc = "Industrial"
    elif risk_score >= 60:
        grade = "C"
        desc = "Consumer"
    else:
        grade = "D"
        desc = ""

    # Cascading MTTF requirements per grade:
    # Grade A: minimum 15 years, Grade B: minimum 10 years, Grade C: minimum 5 years.
    if grade == "A" and mttf_years < 15.0:
        grade = "B"
        desc = "Industrial"
    if grade == "B" and mttf_years < 10.0:
        grade = "C"
        desc = "Consumer"
    if grade == "C" and mttf_years < 5.0:
        grade = "D"
        desc = ""

    if mttf_years < 3.0 or predicted_yield < 0.60:
        grade = "D"
        desc = ""

    corr_value = pd.to_numeric(golden_correlation, errors="coerce")
    if not pd.isna(corr_value) and float(corr_value) < 0.20 and grade != "D":
        grade = "D"
        desc = ""

    return {
        "Grade": grade,
        "Application": desc,
    }


def score_post_packaging_reliability(
    wafer_id: str,
    mttf_years: float,
    golden_correlation: float = 0.0,
    predicted_yield: float = 1.0,
    golden_similarity_probe: float = None,
    golden_similarity_ft: float = None,
    probe_corr: float = None,
    probe_z_closeness: float = None,
    ft_corr: float = None,
    ft_z_closeness: float = None,
    probe_max_z: float = None,
    ft_max_z: float = None,
) -> dict:
    """
    Unified post-packaging scorer that computes both reliability score and
    product grade from MTTF plus golden similarities and correlations.
    """
    score_dict = compute_reliability_score(
        wafer_id=wafer_id,
        mttf_years=mttf_years,
        golden_correlation=golden_correlation,
        predicted_yield=predicted_yield,
        golden_similarity_probe=golden_similarity_probe,
        golden_similarity_ft=golden_similarity_ft,
        probe_corr=probe_corr,
        probe_z_closeness=probe_z_closeness,
        ft_corr=ft_corr,
        ft_z_closeness=ft_z_closeness,
        probe_max_z=probe_max_z,
        ft_max_z=ft_max_z,
    )
    grade_dict = compute_product_grade(
        risk_score=score_dict["Risk_Score"],
        mttf_years=mttf_years,
        golden_correlation=golden_correlation,
        predicted_yield=predicted_yield,
    )

    return {
        "Wafer_ID": wafer_id,
        "Risk_Score": score_dict["Risk_Score"],
        "Grade": grade_dict["Grade"],
        "Application": grade_dict["Application"],
        "Components": score_dict["Components"],
    }
