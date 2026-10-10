from pathlib import Path

from ml_compute_statistic import find_product_limit_file, parse_limit_file


def test_deployed_ddr4_specs_resolve_for_temporary_upload():
    root = Path(__file__).resolve().parents[1] / "dataset"
    path = find_product_limit_file(Path("upload/wafer.csv"), root, "DDR4SDRAM", "MT40A1G8")
    assert path == root / "J750/DDR4SDRAM_MT40A1G8/ddr4_limits.csv"
    assert parse_limit_file(path)["T1.0"] == (1.14, 1.26)


def test_exact_lot_limits_take_priority_over_shared_specs(tmp_path):
    product = tmp_path / "J750/DDR4SDRAM_PART"
    probe = product / "T&P_Decrypted/lot/limit"
    probe.mkdir(parents=True)
    shared = product / "ddr4_limits.csv"
    shared.touch()
    exact = probe / "wafer.std_1_limits.csv"
    exact.touch()
    assert find_product_limit_file(Path("upload/wafer.std_1.csv"), tmp_path, "DDR4SDRAM", "PART") == exact


def test_does_not_use_another_part_or_unrelated_lot_limits(tmp_path):
    wrong = tmp_path / "J750/DDR4SDRAM_OTHER"
    wrong.mkdir(parents=True)
    (wrong / "ddr4_limits.csv").touch()
    probe = tmp_path / "J750/DDR4SDRAM_PART/T&P_Decrypted/limit"
    probe.mkdir(parents=True)
    (probe / "another_wafer_limits.csv").touch()
    assert find_product_limit_file(Path("upload/current.csv"), tmp_path, "DDR4SDRAM", "PART") is None
