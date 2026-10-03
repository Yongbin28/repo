import numpy as np
import pandas as pd

from waferpulse.tools.local_automl import benchmark_local_regression


def test_local_automl_runs_nine_regressors_with_grouped_validation() -> None:
    rng = np.random.default_rng(42)
    rows = 60
    frame = pd.DataFrame(
        {
            "lot": np.repeat([f"lot_{i}" for i in range(6)], 10),
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
        }
    )
    frame["target"] = 2.0 * frame["x1"] - frame["x2"] + rng.normal(0, 0.1, rows)

    metrics, predictions, metadata = benchmark_local_regression(
        frame,
        target_column="target",
        group_column="lot",
        search_profile="conservative",
    )

    assert len(metrics) == 9
    assert metrics.iloc[0]["r2"] > 0.5
    assert predictions["model"].nunique() == 9
    assert "GroupKFold" in metadata["validation"]
